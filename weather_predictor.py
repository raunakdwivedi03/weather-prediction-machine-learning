import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')


class WeatherPredictor:
    """
    Advanced weather prediction model using machine learning.
    
    Supports multiple algorithms: Gradient Boosting, Random Forest, Neural Networks.
    """
    
    def __init__(self, model_name='gradient_boost'):
        """
        Initialize the WeatherPredictor.
        
        Args:
            model_name (str): Choice of 'gradient_boost', 'random_forest', or 'neural_net'
        """
        self.model = None
        self.scaler = StandardScaler()
        self.model_name = model_name
        self.performance_metrics = {}
        self.feature_names = ['humidity', 'pressure', 'wind_speed', 'temperature', 
                             'cloud_cover', 'dew_point', 'visibility']
    
    def generate_realistic_data(self, n_samples=2000):
        """
        Generate realistic weather data with temporal patterns.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated weather data
        """
        np.random.seed(42)
        
        days = np.arange(n_samples)
        seasonal_pattern = 15 * np.sin(2 * np.pi * days / 365)
        
        base_temp = 15 + seasonal_pattern
        temperature = base_temp + np.random.normal(0, 2, n_samples)
        
        humidity = 70 - 0.5 * (temperature - base_temp) + np.random.normal(0, 5, n_samples)
        humidity = np.clip(humidity, 20, 100)
        
        pressure = 1015 + 5 * np.sin(2 * np.pi * days / 30) + np.random.normal(0, 2, n_samples)
        
        wind_speed = 8 + 2 * np.abs(pressure - 1015) / 5 + np.random.normal(0, 1.5, n_samples)
        wind_speed = np.clip(wind_speed, 0, 40)
        
        cloud_cover = 40 + 0.4 * (humidity - 70) + np.random.normal(0, 8, n_samples)
        cloud_cover = np.clip(cloud_cover, 0, 100)
        
        dew_point = temperature - (100 - humidity) / 5 + np.random.normal(0, 1, n_samples)
        
        visibility = 10 - (humidity - 50) / 20 - cloud_cover / 50 + np.random.normal(0, 0.5, n_samples)
        visibility = np.clip(visibility, 0.1, 10)
        
        tomorrow_temp = (
            0.6 * temperature +
            0.15 * (humidity - 70) / 30 +
            0.1 * (pressure - 1015) / 10 +
            0.08 * wind_speed / 10 +
            0.05 * cloud_cover / 100 +
            0.02 * dew_point +
            seasonal_pattern * 0.1 +
            np.random.normal(0, 0.8, n_samples)
        )
        
        df = pd.DataFrame({
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'temperature': temperature,
            'cloud_cover': cloud_cover,
            'dew_point': dew_point,
            'visibility': visibility,
            'tomorrow_temp': tomorrow_temp
        })
        
        return df
    
    def prepare_data(self, df, test_size=0.2):
        """
        Split and scale data for training.
        
        Args:
            df (pd.DataFrame): Input data
            test_size (float): Test set size
            
        Returns:
            tuple: Scaled and original train/test sets
        """
        X = df[self.feature_names]
        y = df['tomorrow_temp']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train(self, X_train, y_train, X_test, y_test):
        """
        Train the selected model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            np.array: Predictions on test set
        """
        if self.model_name == 'gradient_boost':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
        elif self.model_name == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == 'neural_net':
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        self.performance_metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return y_pred
    
    def get_feature_importance(self, X_train):
        """
        Extract feature importance scores.
        
        Args:
            X_train: Training features
            
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        return None
    
    def predict(self, conditions_dict):
        """
        Make prediction for new conditions.
        
        Args:
            conditions_dict (dict): Dictionary with weather variables
            
        Returns:
            float: Predicted temperature
        """
        conditions = np.array([[
            conditions_dict['humidity'],
            conditions_dict['pressure'],
            conditions_dict['wind_speed'],
            conditions_dict['temperature'],
            conditions_dict['cloud_cover'],
            conditions_dict['dew_point'],
            conditions_dict['visibility']
        ]])
        
        conditions_scaled = self.scaler.transform(conditions)
        prediction = self.model.predict(conditions_scaled)
        
        return prediction[0]
    
    def save_model(self, filepath='weather_model.pkl'):
        """
        Save model and scaler to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='weather_model.pkl'):
        """
        Load saved model and scaler.
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Weather Prediction ML Model")
    print("For usage examples, see example_usage.py")
