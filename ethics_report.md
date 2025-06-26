## Bias Analysis & Mitigation

### MNIST Model Risks:
1. **Digit Representation Bias**: 
   - Left-handed writers underrepresented in training data
   - *Solution*: Augment data with rotations (±15°) and use TF Fairness Indicators to monitor per-digit accuracy

### Amazon Reviews Risks:
1. **Brand Frequency Bias**:
   - Popular brands (Apple/Samsung) dominate sentiment analysis
   - *Solution*: Implement spaCy rule-based oversampling for rare brands

### Debugged TensorFlow Code:
```python
# Fixed dimension mismatch in CNN:
Original: layers.Dense(64, activation='relu')  
Fixed: layers.Flatten() before Dense layers

# Fixed loss function:
Original: loss='binary_crossentropy' 
Fixed: loss='sparse_categorical_crossentropy'  # For integer labels