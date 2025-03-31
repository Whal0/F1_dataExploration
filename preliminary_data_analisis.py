import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastf1.core import Telemetry

#Konfiguracja środowiska
fastf1.Cache.enable_cache('f1_cache')
sns.set_theme(style="whitegrid")

#Pobieranie danych
session = fastf1.get_session(2023, 1, 'Race')
session.load(telemetry = True)

#1. Info o sesji
print(f"Wyścig: {session.event.EventName}")
print(f"Data: {session.event.EventDate}")
print(f"Tor: {session.event.Location}")

#2. Lista kierowców
drivers = session.drivers
drivers_info = []
for driver_code in drivers:
    driver = session.get_driver(driver_code)
    drivers_info.append({
        'Number': driver.DriverNumber,
        'Driver': f"{driver.FirstName} {driver.LastName}",
        'Team': driver.TeamName
    })

driver_df = pd.DataFrame(drivers_info)
print("\nKierowcy uczestniczący w wyścigu:")
print(driver_df)

#3. okrążenia
laps = session.laps

# Średni czas okrążenia na zestawie opon
compound_lap_times = laps[['Compound', 'LapTime']].copy()
compound_lap_times['LapTime'] = compound_lap_times['LapTime'].dt.total_seconds()
mean_lap_times = compound_lap_times.groupby('Compound').mean().reset_index()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Compound', y='LapTime', data=compound_lap_times)
plt.title('Rozkład czasów okrążeń w zależności od typu opon')
plt.ylabel('Czas okrążenia (s)')
plt.xlabel('Typ opony')
plt.show()

#4. analiza prędkości

all_car_data = []
for driver_code in session.drivers:
    driver_data = session.car_data[str(driver_code)].reset_index()
    driver_data['DriverNumber'] = str(driver_code)
    all_car_data.append(driver_data)

car_data = pd.concat(all_car_data)

#agregacja danych z częstotliwością 1 sekundy
speed_analysis = (car_data
                  .groupby(['DriverNumber', pd.Grouper(key='Time', freq='1s')])['Speed']
                  .mean()
                  .reset_index())

plt.figure(figsize=(12, 6))
for driver_code in session.drivers[:5]: #top5 kierowcow
    driver_name = driver_df.loc[driver_df['Number'] == driver_code, 'Driver'].values[0]
    driver_speed = speed_analysis[speed_analysis['DriverNumber'] == str(driver_code)]
    plt.plot(driver_speed['Time'], driver_speed['Speed'], label=driver_name)

plt.title('Zmiany prędkości podczas wyścigu')
plt.ylabel('Prędkość (km/h)')
plt.xlabel('Czas wyścigu')
plt.legend()
plt.tight_layout()
plt.show()

#5. zużycie opon
stint_data = laps[['Driver', 'Stint', 'Compound', 'LapNumber']].drop_duplicates()
stint_data = stint_data.groupby(['Driver', 'Stint', 'Compound'])['LapNumber'].count().reset_index()
stint_data.rename(columns={'LapNumber': 'LapCount'}, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Driver', y='LapCount', hue='Compound', data=stint_data)
plt.title('Długość stintów w zależności od kierowcy i typu opon')
plt.ylabel('Liczba okrążeń')
plt.xticks(rotation=45)
plt.show()

#7. pogoda
weather_data = session.weather_data
plt.figure(figsize=(12, 6))
weather_data['Time'] = weather_data['Time'].dt.total_seconds() / 60  # Konwersja na minuty

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

ax1.plot(weather_data['Time'], weather_data['AirTemp'], 'b-', label='Temperatura powietrza')
ax2.plot(weather_data['Time'], weather_data['TrackTemp'], 'r-', label='Temperatura toru')

ax1.set_xlabel('Czas wyścigu (minuty)')
ax1.set_ylabel('Temperatura powietrza (°C)', color='b')
ax2.set_ylabel('Temperatura toru (°C)', color='r')
plt.title('Zmiany temperatury podczas wyścigu')
plt.show()

# #8. analiza pozycji  - gotowa funkcja w fastf1

#
# plt.figure(figsize=(12, 8))
# for driver_code in session.drivers[:3]:  # Pierwszych 3 kierowców
#     driver_pos = session.pos_data[driver_code]
#     driver_code_str = driver_df.loc[driver_df['Number'] == driver_code, 'Code'].values[0]
#     plt.plot(driver_pos['X'], driver_pos['Y'], label=driver_code_str)
#
# plt.title('Pozycje bolidów na torze')
# plt.xlabel('Pozycja X')
# plt.ylabel('Pozycja Y')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()