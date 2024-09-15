from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def train_model(X_train, y_train):
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
    
    # Fit the model with progress tracking
    for i in tqdm(range(1, 101), desc="Training Progress"):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
    
    return model