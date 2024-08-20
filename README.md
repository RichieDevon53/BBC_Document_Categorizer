# BBC News Classification Using LSTM
### Overview
This project involves the development of a machine learning model to classify BBC news articles into different categories such as sports, entertainment, technology, business, and politics. The model leverages deep learning techniques, specifically a Long Short-Term Memory (LSTM) network, to achieve high accuracy in predicting the category of a given news article based on its content. <a href='https://huggingface.co/spaces/53Devon/BBC_news_categorizer'>Deployment Link</a>

### Dataset
The dataset used in this project consists of BBC news articles that are pre-labeled into five categories: sports, entertainment, technology, business, and politics. The dataset is relatively balanced across categories, making it suitable for supervised learning.

### Project Structure
```bash
├── bbc-text.csv # The dataset used for training and testing
├── Modelling.ipynb # Jupyter Notebook with the complete analysis and model
├── Sample_inference.ipynb # Example the using of the Model
├── function.py # Personal function for efficiency
└── README.md # This README file
``` 

### Usage
To train the model and make predictions on new data, follow the steps below:
1. **Preprocess the Data:**
    - Tokenize and vectorize the text data using TensorFlow’s TextVectorization layer.
2. **Train the Model:**
    - The model architecture is an LSTM network with four hidden layers of sizes 64, 128, 128, and 64 neurons.
    - The model is trained using a standard deep learning framework with an emphasis on accuracy and generalization.
3. **Evaluate the Model:**
    - The model is evaluated on a separate test set and achieves an accuracy of 95%.
    - Additional metrics like loss, precision, recall, and F1-score can also be evaluated.

### Model Performance
- Accuracy: 95%
- The model shows strong performance in classifying news articles into the correct categories, with sports articles being the most accurately classified.

### Conclusion
- The LSTM model developed in this project is highly effective for the classification of BBC news articles.
- Despite the longer training times associated with LSTM networks compared to simpler models like SimpleRNN, the accuracy gains justify the choice.

### Contributing
Contributions to this project are welcome. If you have suggestions for improvements or find any issues, feel free to create a pull request or open an issue.


