import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class PowerOptimizationModel:
    def __init__(self):
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = MinMaxScaler()
        self.feature_names = [
            'household_size', 'house_size_sqft', 'hvac_usage_hours', 
            'appliance_rating_kw', 'temperature_f', 'time_of_day',
            'seasonal_factor', 'co2_intensity'
        ]
        self.target_names = ['optimal_power_kwh', 'estimated_cost_usd', 'co2_emissions_lbs']
        
        # Historical CO2 data from provided CSV
        self.co2_data = [
            72.076, 64.442, 64.084, 60.842, 61.798, 66.538, 72.626, 75.181,
            70.55, 62.929, 64.519, 60.544, 64.687, 64.736, 73.698, 72.559,
            72.708, 65.117, 66.532, 61.975, 62.031, 67.875, 74.184, 77.029,
            81.264, 71.058, 73.361, 68.703, 68.985, 73.936, 80.968, 81.962,
            87.215, 75.943, 75.092, 68.641, 74.916, 80.942, 90.667, 89.046,
            85.191, 71.476, 67.829, 69.051, 74.201, 81.372, 88.003, 91.836,
            93.946, 83.91, 83.689, 78.077, 83.19, 88.15, 96.579, 97.246,
            100.568, 94.862, 93.21, 81.245, 82.785, 91.484, 107.126, 106.245,
            108.164, 94.766, 95.724, 86.386, 89.002, 99.05, 111.044, 107.758,
            113.136, 97.313, 95.335, 86.587, 90.83, 94.419, 109.995, 109.179,
            105.965, 90.912, 93.568, 86.575, 90.75, 99.98, 119.945, 126.652,
            119.436, 103.635, 108.149, 94.329, 98.181, 112.983, 119.701, 125.724,
            125.429, 109.359, 107.966, 100.317, 107.595, 113.584, 126.626, 124.309,
            126.887, 109.086, 106.803, 95.341, 101.892, 116.698, 134.789, 122.28,
            124.043, 106.754, 108.607, 102.222, 112.257, 126.2, 140.582, 139.268,
            133.468, 120.783, 115.584, 106.49, 111.2, 128.537, 141.139, 148.031,
            132.233, 124.345, 122.803, 111.194, 115.915, 126.031, 138.083, 139.614,
            132.923, 116.26, 121.806, 115.909, 118.556, 130.719, 143.146, 146.257,
            143.148, 117.505, 118.823, 111.147, 122.745, 131.945, 144.042, 144.107,
            137.39, 121.125, 126.201, 118.33, 121.946, 129.054, 148.563, 142.166,
            139.269, 128.056, 134.599, 119.596, 120.469, 136.697, 157.967, 156.392,
            153.119, 131.25, 132.54, 120.392, 126.496, 146.639, 153.375, 151.757,
            143.097, 127.774, 127.348, 118.414, 125.516, 138.913, 159.638, 167.717,
            153.915, 138.446, 138.362, 124.885, 135.014, 147.23, 160.939, 162.998,
            162.245, 135.873, 138.495, 130.325, 136.525, 147.624, 169.108, 164.925,
            159.602, 138.682, 144.143, 133.418, 146.249, 158.591, 174.995, 174.745,
            161.387, 138.081, 145.169, 137.504, 144.891, 157.377, 179.42, 173.707,
            172.903, 155.75, 152.347, 137.295, 151.232, 165.771, 172.816, 179.476,
            171.379, 146.187, 151.068, 136.46, 148.784, 159.457, 176.898, 180.673,
            159.356, 139.482, 146.594, 138.353, 147.956, 161.561, 179.351, 177.391,
            175.518, 152.63, 150.919, 138.322, 147.491, 160.056, 178.656, 181.673,
            175.35, 157.548, 149.42, 138.155, 153.824, 164.022, 178.541, 175.913,
            175.514, 153.555, 159.857, 140.548, 151.645, 171.133, 184.087, 186.022,
            166.165, 154.226, 156.997, 138.082, 153.221, 166.169, 184.807, 186.844,
            174.337, 159.772, 155.545, 144.038, 154.284, 170.559, 183.582, 188.407,
            177.865, 163.114, 156.693, 144.524, 152.626, 168.239, 184.592, 179.993,
            169.01, 138.457, 134.157, 125.161, 131.317, 147.22, 157.083, 161.543,
            169.541, 149.732, 142.659, 125.288, 141.697, 163.128, 177.165, 176.655,
            166.309, 135.738, 133.862, 123.65, 135.488, 155.016, 173.728, 169.987,
            129.542, 115.247, 105.084, 94.674, 114.96, 131.167, 158.331, 151.36,
            137.055, 122.562, 128.606, 110.959, 118.029, 137.026, 151.952, 149.701,
            153.85, 140.102, 132.664, 106.751, 117.672, 136.577, 149.776, 148.995,
            130.232, 122.417, 106.412, 88.646, 104.498, 126.28, 140.283, 135.156,
            113.495, 92.416, 72.84, 71.41, 82.51, 115.772, 135.958
        ]
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic training data based on realistic household patterns"""
        np.random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            # Household characteristics
            household_size = np.random.randint(1, 8)
            house_size = np.random.normal(2200, 800)  # sq ft
            house_size = max(500, min(8000, house_size))
            
            # Usage patterns
            hvac_usage = np.random.uniform(0, 20)  # hours per day
            appliance_rating = np.random.uniform(2, 20)  # kW
            temperature = np.random.normal(72, 25)  # Fahrenheit
            temperature = max(-10, min(110, temperature))
            time_of_day = np.random.randint(0, 4)  # 0=night, 1=morning, 2=afternoon, 3=evening
            
            # Seasonal factor (simulate yearly cycle)
            day_of_year = np.random.randint(1, 366)
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * day_of_year / 365)
            
            # CO2 intensity (use historical data)
            co2_intensity = np.random.choice(self.co2_data)
            
            # Calculate optimal power consumption
            base_power = household_size * 1.5 + (house_size / 1000) * 2.5
            hvac_power = hvac_usage * abs(temperature - 72) / 20 * 1.8
            appliance_power = appliance_rating * np.random.uniform(0.4, 0.8)
            
            # Time-of-day multipliers
            time_multipliers = [0.6, 0.9, 1.3, 1.1]  # night, morning, afternoon, evening
            time_factor = time_multipliers[time_of_day]
            
            # Total optimal power
            optimal_power = (base_power + hvac_power + appliance_power) * time_factor * seasonal_factor
            optimal_power = max(1.0, optimal_power)  # Minimum 1 kWh
            
            # Calculate cost (varies by time and demand)
            base_rate = 0.12  # $0.12 per kWh base rate
            peak_rate_multiplier = [0.8, 1.0, 1.4, 1.2][time_of_day]  # peak pricing
            demand_charge = 0.02 if optimal_power > 15 else 0  # demand charge for high usage
            estimated_cost = optimal_power * base_rate * peak_rate_multiplier + demand_charge
            
            # Calculate CO2 emissions
            co2_factor = co2_intensity / 1000  # convert to appropriate units
            co2_emissions = optimal_power * co2_factor * 2.2  # lbs CO2 per kWh
            
            data.append([
                household_size, house_size, hvac_usage, appliance_rating, 
                temperature, time_of_day, seasonal_factor, co2_intensity,
                optimal_power, estimated_cost, co2_emissions
            ])
        
        df = pd.DataFrame(data, columns=self.feature_names + self.target_names)
        return df
    
    def build_model(self):
        """Build neural network model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(len(self.target_names), activation='linear')
        ])
        
        # Custom loss function that penalizes both cost and CO2
        def multi_objective_loss(y_true, y_pred):
            power_loss = tf.keras.losses.mse(y_true[:, 0], y_pred[:, 0])
            cost_loss = tf.keras.losses.mse(y_true[:, 1], y_pred[:, 1]) * 10  # weight cost more
            co2_loss = tf.keras.losses.mse(y_true[:, 2], y_pred[:, 2]) * 5   # weight CO2 more
            return power_loss + cost_loss + co2_loss
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=multi_objective_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, data=None, epochs=150, batch_size=64, validation_split=0.2):
        """Train the power optimization model"""
        print(" Generating training data...")
        
        if data is None:
            data = self.generate_synthetic_data()
        
        # Separate features and targets
        X = data[self.feature_names].values
        y = data[self.target_names].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features and targets
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)
        y_train_scaled = self.scaler_targets.fit_transform(y_train)
        y_test_scaled = self.scaler_targets.transform(y_test)
        
        print("🧠 Building neural network...")
        self.model = self.build_model()
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        print("🚀 Training model...")
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        print("\n📊 Evaluating model performance...")
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_targets.inverse_transform(y_pred_scaled)
        
        for i, target in enumerate(self.target_names):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            print(f"{target}: MSE={mse:.4f}, MAE={mae:.4f}")
        
        return history
    
    def predict(self, household_size, house_size, hvac_usage, appliance_rating, 
                temperature, time_of_day, seasonal_factor=1.0):
        """Make prediction for optimal power consumption"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Use average CO2 intensity if not specified
        co2_intensity = np.mean(self.co2_data)
        
        # Prepare input
        input_data = np.array([[
            household_size, house_size, hvac_usage, appliance_rating,
            temperature, time_of_day, seasonal_factor, co2_intensity
        ]])
        
        # Scale input
        input_scaled = self.scaler_features.transform(input_data)
        
        # Make prediction
        prediction_scaled = self.model.predict(input_scaled, verbose=0)
        prediction = self.scaler_targets.inverse_transform(prediction_scaled)
        
        result = {
            'optimal_power_kwh': float(prediction[0][0]),
            'estimated_cost_usd': float(prediction[0][1]),
            'co2_emissions_lbs': float(prediction[0][2]),
            'efficiency_score': min(95, max(60, 100 - (prediction[0][2] / prediction[0][0]) * 50))
        }
        
        return result
    
    def optimize_power_supply(self, household_params, time_periods=24):
        """Optimize power supply for different time periods"""
        results = []
        
        for hour in range(time_periods):
            time_of_day = hour // 6  # 0-5: night, 6-11: morning, 12-17: afternoon, 18-23: evening
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (hour % 365) / 365)
            
            prediction = self.predict(
                household_params['household_size'],
                household_params['house_size'],
                household_params['hvac_usage'],
                household_params['appliance_rating'],
                household_params['temperature'],
                time_of_day,
                seasonal_factor
            )
            
            prediction['hour'] = hour
            results.append(prediction)
        
        return results
    
    def save_model(self, filepath='power_optimization_model'):
        """Save trained model and scalers"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model
        self.model.save(f'{filepath}.h5')
        
        # Save scalers
        joblib.dump(self.scaler_features, f'{filepath}_feature_scaler.pkl')
        joblib.dump(self.scaler_targets, f'{filepath}_target_scaler.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'co2_data_stats': {
                'mean': float(np.mean(self.co2_data)),
                'std': float(np.std(self.co2_data)),
                'min': float(np.min(self.co2_data)),
                'max': float(np.max(self.co2_data))
            }
        }
        
        with open(f'{filepath}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='power_optimization_model'):
        """Load trained model and scalers"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f'{filepath}.h5')
            
            # Load scalers
            self.scaler_features = joblib.load(f'{filepath}_feature_scaler.pkl')
            self.scaler_targets = joblib.load(f'{filepath}_target_scaler.pkl')
            
            # Load metadata
            with open(f'{filepath}_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.target_names = metadata['target_names']
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    def perform_eda(self, data: pd.DataFrame):
        """
        Performs Exploratory Data Analysis (EDA) on the synthetic data.
        Visualizes distributions, correlations, and relationships between features and targets.
        """
        print("\n📊 Performing Exploratory Data Analysis (EDA)...")
        print("=" * 50)

        # 1. Display Basic Statistics
        print("\n--- Basic Data Statistics ---")
        print(data.describe())

        # 2. Check for Missing Values (should be none in synthetic data, but good practice)
        print("\n--- Missing Values ---")
        print(data.isnull().sum())

        # 3. Visualize Feature Distributions
        print("\n--- Feature Distributions ---")
        plt.figure(figsize=(18, 12))
        for i, feature in enumerate(self.feature_names):
            plt.subplot(3, 3, i + 1)
            if data[feature].nunique() < 10 and feature not in ['seasonal_factor', 'temperature_f', 'hvac_usage_hours', 'appliance_rating_kw']: # Categorical-like features
                sns.countplot(x=feature, data=data)
            else:
                sns.histplot(data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count / Density')
        plt.tight_layout()
        plt.suptitle('Distribution of Features', y=1.02, fontsize=16)
        plt.show()

        # 4. Visualize Target Distributions
        print("\n--- Target Distributions ---")
        plt.figure(figsize=(15, 5))
        for i, target in enumerate(self.target_names):
            plt.subplot(1, 3, i + 1)
            sns.histplot(data[target], kde=True, color='skyblue')
            plt.title(f'Distribution of {target}')
            plt.xlabel(target)
            plt.ylabel('Density')
        plt.tight_layout()
        plt.suptitle('Distribution of Target Variables', y=1.02, fontsize=16)
        plt.show()

        # 5. Correlation Heatmap
        print("\n--- Correlation Heatmap ---")
        plt.figure(figsize=(12, 10))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Features and Targets')
        plt.show()

        # 6. Feature-Target Relationships (Scatter Plots/Box Plots for key relationships)
        print("\n--- Feature-Target Relationships ---")
        
        # Example: household_size vs. optimal_power_kwh
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.boxplot(x='household_size', y='optimal_power_kwh', data=data)
        plt.title('Optimal Power by Household Size')
        plt.xlabel('Household Size')
        plt.ylabel('Optimal Power (kWh)')

        # Example: temperature_f vs. optimal_power_kwh
        plt.subplot(1, 3, 2)
        sns.scatterplot(x='temperature_f', y='optimal_power_kwh', data=data, alpha=0.6)
        plt.title('Optimal Power vs. Temperature')
        plt.xlabel('Temperature (°F)')
        plt.ylabel('Optimal Power (kWh)')

        # Example: time_of_day vs. estimated_cost_usd
        plt.subplot(1, 3, 3)
        sns.boxplot(x='time_of_day', y='estimated_cost_usd', data=data)
        plt.title('Estimated Cost by Time of Day')
        plt.xlabel('Time of Day (0=Night, 1=Morning, 2=Afternoon, 3=Evening)')
        plt.ylabel('Estimated Cost ($)')
        plt.tight_layout()
        plt.show()

        # Example: co2_intensity vs. co2_emissions_lbs
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='co2_intensity', y='co2_emissions_lbs', data=data, alpha=0.6)
        plt.title('CO2 Emissions vs. CO2 Intensity')
        plt.xlabel('CO2 Intensity')
        plt.ylabel('CO2 Emissions (lbs)')
        plt.show()

        print("\n--- EDA Complete ---")
        print("=" * 50)

# Example usage and testing
if __name__ == "__main__":
    # Initialize and train model
    optimizer = PowerOptimizationModel()
    
    print("🏠 Power Optimization Model Training")
    print("=" * 50)
    
    synthetic_data = optimizer.generate_synthetic_data()

    # Perform EDA before training (or after, to check the generated data)
    optimizer.perform_eda(synthetic_data) # <-- This is the new call

    # Train the model using the generated data
    history = optimizer.train(data=synthetic_data)
    
    # Save the model
    optimizer.save_model()
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    test_params = {
        'household_size': 4,
        'house_size': 2500,
        'hvac_usage': 10,
        'appliance_rating': 8,
        'temperature': 75,
        'time_of_day': 2  # afternoon
    }
    
    result = optimizer.predict(**test_params)
    print("\n📊 Optimization Results:")
    print(f"Optimal Power: {result['optimal_power_kwh']:.2f} kWh")
    print(f"Estimated Cost: ${result['estimated_cost_usd']:.2f}")
    print(f"CO2 Emissions: {result['co2_emissions_lbs']:.2f} lbs")
    print(f"Efficiency Score: {result['efficiency_score']:.1f}%")
    
    # Test 24-hour optimization
    print("\n🕐 24-Hour Optimization...")
    household_params = {
        'household_size': 3,
        'house_size': 2000,
        'hvac_usage': 8,
        'appliance_rating': 6,
        'temperature': 72
    }
    
    daily_results = optimizer.optimize_power_supply(household_params)
    total_power = sum(r['optimal_power_kwh'] for r in daily_results)
    total_cost = sum(r['estimated_cost_usd'] for r in daily_results)
    total_co2 = sum(r['co2_emissions_lbs'] for r in daily_results)
    
    print(f"Daily Total - Power: {total_power:.1f} kWh, Cost: ${total_cost:.2f}, CO2: {total_co2:.1f} lbs")
    print("\n✅ Model training and testing completed successfully!")