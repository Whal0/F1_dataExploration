import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np #nie masz czasu bawic sie w optymalizje
import matplotlib.pyplot as plt
import optuna
import seaborn as sns

global_df = None

def load_data():
    global global_df
    fastf1.Cache.enable_cache('cache')
    session = fastf1.get_session(2023, 'Bahrain', 'R')
    session.load(telemetry=True)
    laps = session.laps
    df = pd.DataFrame(laps)
    global_df = df.copy()
    df = df.dropna(subset=['LapTime', 'LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL'])
    df['LapTime_sec'] = df['LapTime'].dt.total_seconds()
    X = df[['LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL']]
    y = df['LapTime_sec']
    return X, y


X, y = load_data()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=12)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear:.3f}")

# Decision Tree
tree_model = DecisionTreeRegressor(random_state=12)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree MSE: {mse_tree:.3f}")


# Optuna
def objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    tree = DecisionTreeRegressor(
        random_state=12,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


print("\nOptimizing Decision Tree with Optuna...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best parameters:", study.best_params)
print(f"Best Decision Tree MSE: {study.best_value:.3f}")

# Retrain with best params
best_tree = DecisionTreeRegressor(random_state=12, **study.best_params)
best_tree.fit(X_train, y_train)
y_pred_best_tree = best_tree.predict(X_test)

# Example prediction
example = X_test[0]
print("\nExample input (scaled):")
print(example)

pred_linear = linear_model.predict([example])[0]
pred_tree = tree_model.predict([example])[0]
print(f"\nLinear Regression predicted lap time: {pred_linear:.3f} sec")
print(f"Decision Tree predicted lap time: {pred_tree:.3f} sec")
print(f"Actual lap time: {y_test.iloc[0]:.3f} sec")



results_df = X.copy()
results_df['LapNumber'] = X['LapNumber']
results_df['Actual'] = y.values
results_df['Predicted_Linear'] = linear_model.predict(scaler.transform(X))
results_df['Predicted_Tree'] = tree_model.predict(scaler.transform(X))
results_df['Predicted_OptunaTree'] = best_tree.predict(scaler.transform(X))

# srednie z okrazen
agg_df = results_df.groupby('LapNumber').agg({
    'Actual': 'mean',
    'Predicted_Linear': 'mean',
    'Predicted_Tree': 'mean',
    'Predicted_OptunaTree': 'mean'
}).reset_index()

# Przcinamy po IQR
Q1 = agg_df['Actual'].quantile(0.25)
Q3 = agg_df['Actual'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
mask = (
        (agg_df['Actual'] <= upper_bound) & (agg_df['Actual'] >= lower_bound) &
        (agg_df['Predicted_Linear'] <= upper_bound) & (agg_df['Predicted_Linear'] >= lower_bound) &
        (agg_df['Predicted_Tree'] <= upper_bound) & (agg_df['Predicted_Tree'] >= lower_bound) &
        (agg_df['Predicted_OptunaTree'] <= upper_bound) & (agg_df['Predicted_OptunaTree'] >= lower_bound)
)
agg_clean = agg_df[mask]

# Melt to long format for seaborn
plot_long = pd.melt(
    agg_clean,
    id_vars=['LapNumber'],
    value_vars=['Actual', 'Predicted_Linear', 'Predicted_Tree', 'Predicted_OptunaTree'],
    var_name='Type',
    value_name='LapTime'
)

plt.figure(figsize=(14, 7))
# plot prawda
sns.scatterplot(
    data=plot_long[plot_long['Type'] == 'Actual'],
    x='LapNumber',
    y='LapTime',
    hue='Type',
    palette=['black'],
    s=60,
    alpha=1.0,
    edgecolor='w',
    legend=False
)
# Plot pred
sns.scatterplot(
    data=plot_long[plot_long['Type'] != 'Actual'],
    x='LapNumber',
    y='LapTime',
    hue='Type',
    palette=['blue', 'green', 'orange'],
    s=30,
    alpha=0.4,
    edgecolor='w'
)
plt.xlabel("Lap Number")
plt.ylabel("Lap Time (seconds)")
plt.gca().invert_yaxis()
plt.suptitle("Actual vs Predicted Lap Times per Lap (Linear, Tree, Optuna Tree)")
plt.grid(
    color='lightgray',
    linestyle='--',
    alpha=0.7,
    which='both',
    axis='both'
)
sns.despine(left=False, bottom=False)
plt.tight_layout()
plt.show()

# --

driver1 = 'VER'
driver2 = 'LEC'


filtered = global_df.dropna(subset=['LapTime', 'LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'Driver'])
filtered['LapTime_sec'] = filtered['LapTime'].dt.total_seconds()
filtered = filtered[filtered['Driver'].isin([driver1, driver2])].copy()

# Predict using models trained on the whole dataset
X_drivers = filtered[['LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL']]
X_drivers_scaled = scaler.transform(X_drivers)
filtered['Predicted_Linear'] = linear_model.predict(X_drivers_scaled)
filtered['Predicted_Tree'] = tree_model.predict(X_drivers_scaled)
filtered['Predicted_OptunaTree'] = best_tree.predict(X_drivers_scaled)

# Group by LapNumber and Driver, take mean (in case of duplicates)
df_2drivers_agg = filtered.groupby(['LapNumber', 'Driver']).agg({
    'LapTime_sec': 'mean',
    'Predicted_Linear': 'mean',
    'Predicted_Tree': 'mean',
    'Predicted_OptunaTree': 'mean'
}).reset_index()

# Melt for plotting
plot_2drivers = pd.melt(
    df_2drivers_agg,
    id_vars=['LapNumber', 'Driver'],
    value_vars=['LapTime_sec', 'Predicted_Linear', 'Predicted_Tree', 'Predicted_OptunaTree'],
    var_name='Type',
    value_name='LapTime'
)

plt.figure(figsize=(15, 7))
for driver, color in zip([driver1, driver2], ['blue', 'red']):
    # Actual
    sns.lineplot(
        data=plot_2drivers[(plot_2drivers['Driver'] == driver) & (plot_2drivers['Type'] == 'LapTime_sec')],
        x='LapNumber', y='LapTime', label=f'{driver} Actual', color=color, linewidth=2)
    # Linear
    sns.lineplot(
        data=plot_2drivers[(plot_2drivers['Driver'] == driver) & (plot_2drivers['Type'] == 'Predicted_Linear')],
        x='LapNumber', y='LapTime', label=f'{driver} Linear', color=color, linestyle='--', alpha=0.6)
    # Tree
    sns.lineplot(
        data=plot_2drivers[(plot_2drivers['Driver'] == driver) & (plot_2drivers['Type'] == 'Predicted_Tree')],
        x='LapNumber', y='LapTime', label=f'{driver} Tree', color=color, linestyle=':', alpha=0.6)
    # Optuna Tree
    sns.lineplot(
        data=plot_2drivers[(plot_2drivers['Driver'] == driver) & (plot_2drivers['Type'] == 'Predicted_OptunaTree')],
        x='LapNumber', y='LapTime', label=f'{driver} OptunaTree', color=color, linestyle='-.', alpha=0.6)

plt.xlabel('Lap Number')
plt.ylabel('Lap Time (seconds)')
plt.gca().invert_yaxis()
plt.title(f'Actual vs Predicted Lap Times for {driver1} and {driver2} (Full Dataset Model)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
