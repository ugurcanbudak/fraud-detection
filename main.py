from scripts.data_preprocessing import preprocess_data
from scripts.train import train_model
from scripts.evaluate_model import evaluate_model
import joblib

def main():
    print("Starting the fraud detection pipeline")

    # Preprocess the data
    print("Preprocessing the data")
    X_train, X_test, y_train, y_test = preprocess_data('data/creditcard.csv')

    # Train the model
    print("Training the model")
    model = train_model(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model")
    evaluate_model(model, X_test, y_test)

    # Save the model
    print("Saving the model")
    joblib.dump(model, 'models/fraud_detection_model.pkl')

    print("Fraud detection pipeline completed successfully")

if __name__ == "__main__":
    main()