import os

# Define the folder structure
folder_structure = {
    '.vscode': {
        'settings.json': ''
    },
    '.github': {
        'workflows': {
            'unittests.yml': ''
        }
    },
    '.gitignore': '',
    'requirements.txt': '',
    'README.md': '## Fraud Detection Project\nThis project aims to improve fraud detection for e-commerce and bank transactions.',
    'src': {
        'data_preprocessing.py': '',
        'model_building.py': '',
        'model_evaluation.py': '',
        'model_explainability.py': '',
        'model_deployment.py': '',
        'dashboard.py': ''
    },
    'notebooks': {
        '__init__.py': '',
        'EDA.ipynb': '## Exploratory Data Analysis\nDetailed analysis of the datasets.',
        'Feature_Engineering.ipynb': '## Feature Engineering\nDocumenting feature creation processes.',
        'Model_Training.ipynb': '## Model Training\nTraining various models and evaluating their performance.',
        'Model_Explainability.ipynb': '## Model Explainability\nUsing SHAP and LIME for model interpretation.',
        'Model_Deployment.ipynb': '## Model Deployment\nDeploying the trained models using Flask and Docker.'
    },
    'tests': {
        '__init__.py': '',
        'test_data_preprocessing.py': '',
        'test_model_building.py': '',
        'test_model_explainability.py': '',
        'test_model_deployment.py': ''
    },
    'scripts': {
        '__init__.py': '',
        'data_analysis.py': '',
        'feature_engineering.py': '',
        'training_pipeline.py': '',
        'api.py': '',
        'docker_setup.py': ''
    },
    'dashboard': {
        '__init__.py': '',
        'dashboard_layout.py': '',
        'visualizations.py': ''
    },
    'logs': {
        'app.log': '## Log file for tracking application behavior and errors.'
    }
}

def create_structure(base_path, structure):
    for name, contents in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(contents, dict):
            # Create directory
            os.makedirs(path, exist_ok=True)
            # Recursively create subdirectories and files
            create_structure(path, contents)
        else:
            # Create file with optional content
            with open(path, 'w') as f:
                f.write(contents)

# Create the folder structure starting from the current directory
create_structure('.', folder_structure)

