import tkinter as tk
from src.network.model_prediction import predict_sentiment

class App:
    SENTIMENTS = ['Negative', 'Neutral', 'Positive']

    def __init__(self, model, vocabulary, max_length):
        self.root = tk.Tk()
        self.root.geometry('500x375')
        self.root.title('Sentiment analysis')

        self.model = model
        self.vocab = vocabulary
        self.max_length = max_length
        self.font = ('Roboto', 12)

        self.title_label = tk.Label(self.root, text = 'Type your sentence: ', font = ('Roboto', 14))
        self.title_label.pack(pady = 5)

        # text area
        self.text_entry = tk.Text(self.root, font = self.font, width = 45, height = 10)
        self.text_entry.pack(pady = 10, padx = 10)

        # receives the input and predicts the sentiment using pretrained LSTM model
        self.submit_button = tk.Button(self.root, text = 'Check sentiment', font = self.font,
                                       command = self.display_sentiment)
        self.submit_button.pack(pady=5)

        # clears the text area
        self.clear_button = tk.Button(self.root, text = 'Clear', font = self.font,
                                      command = self.clear)
        self.clear_button.pack(pady=5)

        # displays the predicted sentiment
        self.result_label = tk.Label(self.root, font = self.font)
        self.result_label.pack(pady=5)

        self.root.mainloop()

    def clear(self):
        """
        Clears the text area, so the user can type in new sentence
        """
        self.text_entry.delete(1.0, tk.END)

    def display_sentiment(self):
        """
        Receives the user input from the text area, predicts its sentiment and displays it
        """
        sentence = self.text_entry.get(1.0, tk.END)

        predicted = predict_sentiment(self.model, sentence, App.SENTIMENTS, self.max_length, self.vocab)
        self.result_label['text'] = predicted