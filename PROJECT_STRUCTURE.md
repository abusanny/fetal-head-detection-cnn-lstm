# Project Structure

This document describes the organization and purpose of files and directories in the fetal head detection project.

## Root Level Files

```
fetal-head-detection-cnn-lstm/
├── README.md                  # Main project documentation
├── PROJECT_STRUCTURE.md       # This file
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules
└── LICENSE                    # MIT License
```

### README.md
Comprehensive project overview including:
- Project description and key features
- Dataset information
- Model architecture details
- Installation and usage instructions
- Training configuration
- Results and performance metrics
- Technologies used
- Citation information

### requirements.txt
Lists all Python packages needed to run the project:
- tensorflow>=2.13.0
- keras>=2.13.0
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- opencv-python
- matplotlib, seaborn for visualization

### .gitignore
Ignores unnecessary files:
- __pycache__ directories
- .pyc files
- Virtual environment directories
- Model checkpoint files
- Data directories
- IDE configuration files

## Source Code (src/)

```
src/
├── __init__.py                # Python package marker
├── data_processing.py         # Data loading and preprocessing
├── model.py                   # Model architecture
├── train.py                   # Training pipeline
└── utils.py                   # Utility functions
```

### data_processing.py
**Purpose**: Image and data handling

**Key Functions**:
- `load_excel_labels()`: Load image labels from Excel files
- `preprocess_image()`: Image cropping, padding, normalization
- `normalize_columns()`: Excel data standardization
- `attach_paths()`: Associate image paths with labels
- `split_by_folder()`: Train/validation/test splitting
- `make_sequences()`: Create temporal sequences from images

**Key Classes**:
- `SeqLoader`: Keras Sequence for efficient batch loading

**Outputs**:
- Preprocessed image sequences
- Train/validation/test datasets
- Label mappings

### model.py
**Purpose**: Deep learning model definition

**Architecture Components**:
1. **Input Layer**: Accepts temporal sequences (Batch, 8, 224, 224, 3)
2. **VGG19 Feature Extraction**: TimeDistributed convolutional backbone
3. **LSTM Layers**: Two stacked LSTM layers for temporal modeling
4. **Classification Head**: TimeDistributed dense layers for many-to-many prediction

**Key Functions**:
- `build_model()`: Constructs the CNN-LSTM architecture
- Model compilation with focal loss and metrics

### train.py
**Purpose**: Training pipeline orchestration

**Key Components**:
- Data loading and preparation
- Model initialization
- Loss function configuration (Binary Focal Loss)
- Optimizer setup (Adam)
- Training loop with callbacks:
  - ModelCheckpoint: Saves best model
  - EarlyStopping: Prevents overfitting
- Evaluation on validation/test sets
- Performance metric calculation
- Results visualization and saving

### utils.py
**Purpose**: Shared utility functions

**Possible Functions**:
- `calculate_metrics()`: Confusion matrix, precision, recall, F1
- `plot_confusion_matrix()`: Visualization helper
- `save_results()`: Export metrics and logs
- `load_model()`: Load saved checkpoints
- `normalize_image()`: Standard image normalization

## Notebooks (notebooks/)

```
notebooks/
└── head_not_head_detection.ipynb  # Main Jupyter notebook
```

### head_not_head_detection.ipynb
**Purpose**: Interactive development and experimentation

**Sections**:
1. Environment setup and imports
2. Data loading and exploration
3. Data preprocessing demonstrations
4. Model architecture definition
5. Training execution
6. Results visualization
7. Performance metrics and analysis
8. Hyperparameter tuning results
9. Per-scan performance breakdown

## Data Directories (Not in Repo)

These directories are typically managed locally or on secure servers:

```
PhantomData72Scan/              # Raw ultrasound data
├── Scan5S/
│   └── ImageData/
├── Scan10S/
│   └── ImageData/
├── Scan15S/
│   └── ImageData/
├── Scan20S/
│   └── ImageData/
├── Scan25S/
│   └── ImageData/
└── Scan30S/
    └── ImageData/
```

## Models Directory (models/)

Stores trained model checkpoints:

```
models/
├── best_model_alpha_0.79.h5   # Best performing model
├── checkpoint_epoch_50.h5     # Latest checkpoint
└── model_weights.h5           # Saved weights only
```

## Results Directory (results/)

Contains training logs and evaluation outputs:

```
results/
├── alpha_0.79/
│   ├── metrics_epoch_50.csv   # Training/validation metrics
│   ├── confusion_matrix.png   # Confusion matrix visualization
│   ├── roc_auc_curve.png      # ROC curve
│   ├── pr_auc_curve.png       # Precision-Recall curve
│   └── classification_report.txt
└── alpha_0.80/
    └── ...
```

## Configuration Directory (config/)

Optional configuration files:

```
config/
└── config.yaml                # Hyperparameters and settings
```

**Example config.yaml**:
```yaml
model:
  img_size: 224
  seq_len: 8
  lstm_units_1: 256
  lstm_units_2: 128
  dropout: 0.4

training:
  batch_size: 8
  epochs: 50
  learning_rate: 1e-5
  optimizer: adam
  
loss:
  type: binary_focal_loss
  alpha: 0.79
  gamma: 2.0
```

## Workflow

### Data Preparation
1. Place raw ultrasound images in `PhantomData72Scan/` directories
2. Prepare Excel label files with image classifications
3. Run `data_processing.py` to:
   - Load labels
   - Normalize column names
   - Attach image paths
   - Create temporal sequences
   - Split into train/validation/test

### Model Training
1. Initialize model using `model.py`
2. Configure training parameters
3. Run training loop:
   - Load batches using SeqLoader
   - Forward pass through CNN-LSTM
   - Calculate focal loss
   - Backward propagation
   - Update weights
4. Save best model based on validation loss
5. Generate evaluation metrics

### Evaluation & Analysis
1. Load best model checkpoint
2. Evaluate on test set
3. Generate:
   - Confusion matrix
   - Classification report
   - ROC-AUC curve
   - PR-AUC curve
4. Analyze per-scan performance
5. Document results

## Key Files Generated During Training

- **Model Checkpoints**: `.h5` files with trained weights
- **Training Logs**: CSV files with epoch-wise metrics
- **Visualizations**: PNG files for curves and matrices
- **Reports**: Text files with classification metrics
- **Predictions**: CSV files with model outputs

## IDE Configuration (Optional)

For VS Code with Python:

```
.vscode/
├── settings.json
├── launch.json
└── extensions.json
```

## Package Structure

The `src/` directory functions as a Python package:

```python
from src.data_processing import SeqLoader, make_sequences
from src.model import build_model
from src.train import train_model
from src.utils import calculate_metrics
```

## Reproducibility

To ensure reproducibility:
- All random seeds are fixed in code
- Model architecture is defined in code
- Hyperparameters are version-controlled
- Data splits are deterministic
- Training logs record all parameters

## Future Enhancements

Potential additions to project structure:
- `tests/` directory for unit tests
- `docs/` directory for detailed documentation
- `scripts/` directory for utility scripts
- `data/` directory for sample/test data
- `experiments/` directory for alternative approaches

---

*Last Updated: December 2025*
