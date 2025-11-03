
from weather_predictor import WeatherPredictor


def main():
    """Main example function"""
    
    print("=" * 70)
    print("WEATHER PREDICTION MODEL - EXAMPLE USAGE")
    print("=" * 70)
    
    # Initialize predictor with Gradient Boosting
    print("\n[1] Initializing Gradient Boosting model...")
    predictor = WeatherPredictor(model_name='gradient_boost')
    
    # Generate synthetic weather data
    print("[2] Generating realistic weather data...")
    df = predictor.generate_realistic_data(n_samples=2000)
    print(f"    Generated {len(df)} samples with {len(predictor.feature_names)} features")
    print(f"    Features: {', '.join(predictor.feature_names)}")
    
    # Prepare data
    print("[3] Preparing and scaling data...")
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.prepare_data(df)
    print(f"    Training samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")
    
    # Train model
    print("[4] Training model...")
    y_pred = predictor.train(X_train, y_train, X_test, y_test)
    print("    Training complete!")
    
    # Display performance metrics
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"RMSE (Root Mean Squared Error): {predictor.performance_metrics['rmse']:.4f}°C")
    print(f"MAE (Mean Absolute Error):      {predictor.performance_metrics['mae']:.4f}°C")
    print(f"R² Score:                       {predictor.performance_metrics['r2']:.4f}")
    print(f"MSE (Mean Squared Error):       {predictor.performance_metrics['mse']:.4f}")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    importance_df = predictor.get_feature_importance(X_train_orig)
    if importance_df is not None:
        print(importance_df.to_string(index=False))
    
    # Sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    print("Actual vs Predicted Temperature on Test Set:")
    print("-" * 70)
    print(f"{'Actual':>10} | {'Predicted':>10} | {'Error':>10} | {'Status':>10}")
    print("-" * 70)
    
    correct = 0
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        
        # Determine if prediction is "good" (error < 1°C)
        status = "✓ Good" if error < 1.0 else "✗ Fair"
        if error < 0.5:
            status = "✓ Excellent"
        
        if error < 1.0:
            correct += 1
        
        print(f"{actual:>10.2f} | {predicted:>10.2f} | {error:>10.2f} | {status:>10}")
    
    accuracy = (correct / min(10, len(y_test))) * 100
    print("-" * 70)
    print(f"Prediction Accuracy (error < 1°C): {accuracy:.1f}%")
    
    # Make prediction for new conditions
    print("\n" + "=" * 70)
    print("PREDICTION FOR NEW WEATHER CONDITIONS")
    print("=" * 70)
    
    new_conditions = {
        'humidity': 65,
        'pressure': 1013,
        'wind_speed': 10,
        'temperature': 15,
        'cloud_cover': 45,
        'dew_point': 8,
        'visibility': 8.5
    }
    
    print("Current Weather Conditions:")
    print("-" * 70)
    for key, value in new_conditions.items():
        unit = get_unit(key)
        print(f"  {key.replace('_', ' ').title():20s}: {value:6.2f} {unit}")
    
    prediction = predictor.predict(new_conditions)
    print("-" * 70)
    print(f"\n🌡️  PREDICTED TOMORROW'S TEMPERATURE: {prediction:.2f}°C\n")
    
    # Save model
    print("=" * 70)
    print("[5] Saving model...")
    predictor.save_model('weather_model.pkl')
    print("=" * 70)
    
    # Example: Try other models
    print("\n" + "=" * 70)
    print("COMPARING DIFFERENT MODELS")
    print("=" * 70)
    
    models = ['gradient_boost', 'random_forest', 'neural_net']
    results = {}
    
    for model_name in models:
        print(f"\nTraining {model_name}...")
        pred = WeatherPredictor(model_name=model_name)
        X_train, X_test, y_train, y_test, _, _ = pred.prepare_data(df)
        pred.train(X_train, y_train, X_test, y_test)
        results[model_name] = pred.performance_metrics['rmse']
        print(f"  RMSE: {pred.performance_metrics['rmse']:.4f}°C")
    
    best_model = min(results, key=results.get)
    print(f"\n✓ Best Model: {best_model} with RMSE: {results[best_model]:.4f}°C")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)


def get_unit(feature_name):
    """Get unit for feature"""
    units = {
        'humidity': '%',
        'pressure': 'hPa',
        'wind_speed': 'km/h',
        'temperature': '°C',
        'cloud_cover': '%',
        'dew_point': '°C',
        'visibility': 'km'
    }
    return units.get(feature_name, '')


if __name__ == "__main__":
    main()
