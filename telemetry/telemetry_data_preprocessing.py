import numpy as np
import math
import fastf1
import pandas as pd


## to co wykomentowane zostawiam na potem bo może mi sie jeszcze przydać

#import requests

# class request_data:
    
#     def __init__(self):
#         self.url = 'https://api.openf1.org/v1'
    
#     def fetch_data(self, url, params=None):
        
#         response = requests.get(url, params=params)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return f"Error while fetching data: {response.status_code}"
        
#     def get_car_data(self, params=None):
    
#         url = f'{self.url}/car_data'

#         return self.fetch_data(url, params)
        
#     def get_laps_data(self, params=None):
        
#         url = f'{self.url}/laps'

#         return self.fetch_data(url, params)
    
#     def get_session_data(self, params=None):
        
#         url = f'{self.url}/sessions'
        
#         return self.fetch_data(url, params)
    
#     def get_location_data(self, params=None):
        
#         url = f'{self.url}/location'
        
#         return self.fetch_data(url, params)
    
# class telemetry_preprocessing:
    
#     def __init__(self):
#          pass
     
    # def data_closest_timestamp(self, time, data):
    
    #     minimal_diff = np.abs(time - data["date"][0])
    #     minimal_index = 0
        
    #     for index, value in data['date'].items():
    #         diff = np.abs(time - value)
    #         if minimal_diff > diff:
    #             minimal_diff = diff
    #             minimal_index = index
            
    #     return minimal_index

    # def join_frames_by_date(self, df1, df2):
    
    #     df1_date = df1["date"]
    #     df2_date = df2["date"]
        
    #     if df1_date[0] > df2_date[0]:
    #         start_index = self.data_closest_timestamp(df1_date[0], df2)
    #         stop_index = df2.shape[0] - start_index 
    #         return df1.iloc[:stop_index].DataFrame.join(df2.iloc[start_index:], lsuffix='_left')
    #     else:
    #         start_index = self.data_closest_timestamp(df2_date[0], df1)
    #         stop_index = df1.shape[0] - start_index
    #         return df1.iloc[start_index:].join(df2.iloc[:stop_index], lsuffix='_left')



## do dzielenia kolumny accelerations na dodatnie i ujemne
def divide_column_by_sign(df : pd.DataFrame, column : str) -> pd.DataFrame:
    
    df[f"positive_{column}"] = df[column].apply(lambda x: 0 if x < 0 else x)
    df[f"negative_{column}"] = df[column].apply(lambda x: 0 if x > 0 else x)
    
    return df       

## funkcja do ładowania danych
# def load_data():
#     global global_df

#     session = fastf1.get_session(2023, 'Bahrain', 'R')
#     session.load(telemetry=True)
#     laps = session.laps

#     global_df = laps.copy()  # zachowaj pełne dane globalnie (opcjonalnie)

#     # Filtrowanie i kopiowanie
#     df = laps.dropna(subset=['LapTime', 'LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL']).copy()
#     df['LapTime_sec'] = df['LapTime'].dt.total_seconds()

#     # Zbieranie telemetrii
#     telemetry_data = []
#     for _, lap in df.iterrows():
#         tel = lap.get_telemetry()
#         tel['DriverNumber'] = lap['DriverNumber']  
#         tel['LapNumber'] = lap['LapNumber']
#         telemetry_data.append(tel)

    
#     telemetry_df = pd.concat(telemetry_data, ignore_index=True)
#     telemetry_df['DriverNumber'] = telemetry_df['DriverNumber'].astype(int)
#     # Przygotowanie X i y
#     X = df[['LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL']]
#     y = df['LapTime_sec']
    
#     return telemetry_df, X, y, df

## klasa dla telemetrii z funkcjami pomocniczymi
class TelemetryProcessing: 

    def __init__(self, data : pd.DataFrame):
        self.data = data

    def normalize_drs(self):

        self.data.DRS = self.data.DRS.apply(lambda x: 1 if x in [10, 12, 14] else 0)
        
        return self

    def calculate_mean_lap_speed(self):

        self.data["mean_lap_speed"] = self.data.groupby(["DriverNumber", "LapNumber"])["Speed"].transform("mean")

        return self

    def compute_accelerations(self):

        computations = telemetry_computations()

        all_lon, all_lat = [], []

        for (driver, lap), group in self.data.groupby(['DriverNumber', 'LapNumber']):
            group = group.sort_values('Time')  # sort before computing derivatives
            lon_, lat_ = computations.compute_accelerations(telemetry=group)
            all_lon.append(lon_)
            all_lat.append(lat_)

        all_lon_series = [pd.Series(arr) for arr in all_lon]
        all_lat_series = [pd.Series(arr) for arr in all_lat]

        self.data['lon_acc'] = pd.concat(all_lon_series, ignore_index=True)
        self.data['lat_acc'] = pd.concat(all_lat_series, ignore_index=True)
    
        self.data['abs_lat_acc'] = self.data['lat_acc'].abs()
        self.data['abs_lon_acc'] = self.data['lon_acc'].abs()

        self.data['sum_lat_acc'] = self.data.groupby(['DriverNumber', 'LapNumber'])['abs_lat_acc'].transform('sum')
        self.data['sum_lon_acc'] = self.data.groupby(['DriverNumber', 'LapNumber'])['abs_lon_acc'].transform('sum')

        return self


    def calculate_lap_progress(self):
        
        self.data['TimeNumberLapTime'] = self.data.groupby(['DriverNumber', 'LapNumber']).cumcount() + 1
        self.data['TimeNumberLapCounts'] = self.data.groupby(['DriverNumber', 'LapNumber'])['LapNumber'].transform('count')

        self.data['LapProgress'] = self.data['TimeNumberLapTime'] / self.data['TimeNumberLapCounts']
        
        return self

    ## mało eleganckie, ale jakos musze wyciągać dane dla pojedyńczego lapa z telemetrii

    def get_single_lap_data(self):
        
        final_df = pd.DataFrame(columns=self.data.columns)

            
        for driver in self.data['DriverNumber'].unique():
            driver_df = self.data[self.data['DriverNumber'] == driver]
            
            laps : pd.DataFrame = int(driver_df['LapNumber'].max())

            for lap in range(1, laps + 1):
                lap_df = driver_df[driver_df['LapNumber'] == lap]
                if not lap_df.empty:
                    final_row = lap_df.iloc[[-1], :]
                    final_df = pd.concat([final_df, final_row], axis=0)

        return final_df

## klasa do obliczania przyśpieszeń
class telemetry_computations:
    
    def __init__(self):
        pass
    
    # credits to https://gist.github.com/TracingInsights/4d3bdeb135a01d7b11e35e5f83f60d6a
    #
    # Define a set of helper functions to perform the computations
    #
    # the t

    @classmethod
    def smooth_derivative(cls, t_in, v_in, method = "centered"):
        #
        # Function to compute a smooth estimation of a derivative.
        # [REF: http://holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/]
        #

        # Configuration
        #
        # Derivative method: two options: 'smooth' or 'centered'. Smooth is more conservative
        # but helps to supress the very noisy signals. 'centered' is more agressive but more noisy

        t = t_in.copy()
        v = v_in.copy()
        epsilon = 1e-9
        
        # (0) Prepare inputs
        # (0.1) Time needs to be transformed to seconds
        if t.dtype == 'timedelta64[ns]':
            t = t_in.apply(lambda x: x.total_seconds()).to_numpy()
 
        t = np.array(t)
        v = np.array(v)

        # (0.1) Assert they have the same size
        assert t.size == v.size

        # (0.2) Initialize output
        dvdt = np.zeros(t.size)

        # (1) Manually compute points out of the stencil

        # (1.1) First point
        dvdt[0] = (v[1] - v[0]) / (t[1] - t[0] + epsilon)

        # (1.2) Second point
        dvdt[1] = (v[2] - v[0]) / (t[2] - t[0] + epsilon)

        # (1.3) Third point
        dvdt[2] = (v[3] - v[1]) / (t[3] - t[1] + epsilon)

        # (1.4) Last points
        n = t.size
        dvdt[n - 1] = (v[n - 1] - v[n - 2]) / (t[n - 1] - t[n - 2] + epsilon)
        dvdt[n - 2] = (v[n - 1] - v[n - 3]) / (t[n - 1] - t[n - 3] + epsilon)
        dvdt[n - 3] = (v[n - 2] - v[n - 4]) / (t[n - 2] - t[n - 4] + epsilon)

        # (2) Compute the rest of the points
        if method == "smooth":
            c = [5.0 / 32.0, 4.0 / 32.0, 1.0 / 32.0]
            for i in range(3, t.size - 3):
                for j in range(1, 4):
                    if (t[i + j] - t[i - j]) == 0:
                        dvdt[i] += 0
                    else:
                        dvdt[i] += (
                            2 * j * c[j - 1] * (v[i + j] - v[i - j]) / (t[i + j] - t[i - j])
                        )
                        
        elif method == "centered":
            for i in range(1, t.size - 1):
                for j in range(1, 4):
                    delta_t = t[i + 1] - t[i - 1]
                    if abs(delta_t) > epsilon:
                        dvdt[i] = (v[i + 1] - v[i - 1]) / delta_t
                    else: # Handle case where t[i+1] == t[i-1]
                        # Option 1: Use forward/backward difference
                        delta_t_fwd = t[i+1] - t[i]
                        delta_t_bwd = t[i] - t[i-1]
                        if abs(delta_t_fwd) > epsilon:
                            dvdt[i] = (v[i+1] - v[i]) / delta_t_fwd
                        elif abs(delta_t_bwd) > epsilon:
                            dvdt[i] = (v[i] - v[i-1]) / delta_t_bwd
                        else:
                            dvdt[i] = 0 # Or NaN if points are truly identical in time
        return dvdt

    def transform_to_pipi(self, input_angle):
        """
        Transforms an angle in radians to the range [-pi, pi].

        Args:
            input_angle: Angle in radians.

        Returns:
            Tuple: (output_angle, revolutions)
                output_angle: Angle wrapped to [-pi, pi].
                revolutions: Number of full revolutions difference.
        """
        pi = math.pi
        two_pi = 2 * pi

        # Simple modulo arithmetic approach
        output_angle = (input_angle + pi) % two_pi - pi

        # Ensure the result is exactly within [-pi, pi] due to potential floating point issues near pi
        if np.isclose(output_angle, pi):
            output_angle = -pi
        elif output_angle < -pi: # Should not happen with modulo, but as safeguard
            output_angle += two_pi
        elif output_angle > pi: # Should not happen with modulo
            output_angle -= two_pi


        # Calculate revolutions based on the wrapped angle
        # Use np.round for robustness against floating point inaccuracies
        revolutions = np.round((input_angle - output_angle) / two_pi)

        return output_angle, int(revolutions)

    def remove_acceleration_outliers(self, acc_in):
        """
        Removes outliers from an acceleration array by replacing them.
        Creates a copy of the input array to avoid modifying the original.

        Args:
            acc_in: NumPy array of acceleration values.

        Returns:
            NumPy array with outliers handled.
        """
        acc = acc_in.copy() # Create a copy to avoid modifying the original array
        acc_threshold_g = 7.5 # Threshold in g's

        n = acc.size
        if n == 0:
            return acc # Return empty array if input is empty

        # Handle first point
        if abs(acc[0]) > acc_threshold_g:
            # Consider clipping instead of setting to 0:
            # acc[0] = np.sign(acc[0]) * acc_threshold_g
            acc[0] = 0.0 # Original logic

        # Handle middle points
        for i in range(1, n - 1):
            if abs(acc[i]) > acc_threshold_g:
                # Consider clipping: acc[i] = np.sign(acc[i]) * acc_threshold_g
                # Consider averaging neighbors: acc[i] = (acc[i-1] + acc[i+1]) / 2 (if acc[i+1] is not outlier)
                acc[i] = acc[i - 1] # Original logic: replace with previous value

        # Handle last point
        if n > 1: # Need at least two points to access acc[-2]
            if abs(acc[-1]) > acc_threshold_g:
                # Consider clipping: acc[-1] = np.sign(acc[-1]) * acc_threshold_g
                acc[-1] = acc[-2] # Original logic: replace with second-to-last value
        elif n == 1 and abs(acc[0]) > acc_threshold_g: # If only one point and it's an outlier
            acc[0] = 0.0 # Re-apply first point logic if needed (already done above)


        return acc


    def compute_accelerations(self, telemetry):
        # --- Input Preparation ---
        time_data = telemetry['Time']
        speed_kmh = np.array(telemetry['Speed'])
        distance = np.array(telemetry['Distance'])
        x_coords = np.array(telemetry['X'])
        y_coords = np.array(telemetry['Y'])

        # Convert speed to m/s
        v_mps = speed_kmh / 3.6
        g = 9.81 # Acceleration due to gravity

        # --- Longitudinal Acceleration ---
        # Calculate dv/dt (acceleration in m/s^2)
        lon_acc_mps2 = self.smooth_derivative(time_data, v_mps)
        # Convert to g's
        lon_acc_g = lon_acc_mps2 / g

        # --- Lateral Acceleration ---
        # Calculate path tangent components dx/ds and dy/ds (unitless)
        # Using distance 's' as the independent variable
        dx_ds = self.smooth_derivative(distance, x_coords)
        dy_ds = self.smooth_derivative(distance, y_coords)

        n_points = dx_ds.size
        if n_points == 0:
             return np.array([]), np.array([]) # Handle empty input

        # Calculate path angle theta (radians) iteratively
        theta = np.zeros(n_points)
        if n_points > 0:
             # Initial angle
             theta[0] = math.atan2(dy_ds[0], dx_ds[0])
             # Integrate angle changes, wrapping correctly
             for i in range(1, n_points): # *** FIXED LOOP START ***
                 # Calculate the angle change from the previous point
                 current_segment_angle = math.atan2(dy_ds[i], dx_ds[i])
                 delta_theta_raw = current_segment_angle - theta[i - 1]
                 # Wrap the change to [-pi, pi] to avoid large jumps
                 delta_theta_wrapped, _ = self.transform_to_pipi(delta_theta_raw)
                 # Add the wrapped change to the previous angle
                 theta[i] = theta[i - 1] + delta_theta_wrapped

        # Calculate curvature kappa = d(theta)/ds (rad/meter)
        kappa = self.smooth_derivative(distance, theta)

        # Calculate lateral acceleration: a_lat = v^2 * kappa (m/s^2)
        lat_acc_mps2 = v_mps * v_mps * kappa
        # Convert to g's
        lat_acc_g = lat_acc_mps2 / g

        # --- Remove Outliers ---
        # Note: remove_acceleration_outliers now returns a copy
        lon_acc_g_clean = self.remove_acceleration_outliers(lon_acc_g)
        lat_acc_g_clean = self.remove_acceleration_outliers(lat_acc_g)

        # --- Return rounded results ---
        return np.round(lon_acc_g_clean, 5), np.round(lat_acc_g_clean, 5)

def test():
    
    session = fastf1.get_session(2023, 'Bahrain', 'R')
    session.load()
    
    telemetry = session.car_data['1']
    telemetry = telemetry.add_distance()

    pos_data = session.pos_data['1'] 
    telemetry = telemetry.merge(pos_data, on='Time', how='left') 
    telemetry.dropna(inplace=True)

    lon, lat = telemetry_computations().compute_accelerations(telemetry=telemetry)
    
    return lon, lat

if __name__ == '__main__':

    lon, lat = test()

    print(lon, lat)