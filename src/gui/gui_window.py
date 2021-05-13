import tkinter as tk
from src.network.model_prediction import predict_sentiment

class App:
    SENTIMENTS = ['Negative', 'Neutral', 'Positive']

    def __init__(self, model, vocabulary, max_length):
        self.root = tk.Tk()
        self.root.geometry('500x500')

        self.model = model
        self.vocab = vocabulary
        self.max_length = max_length

        self.title_label = tk.Label(self.root, text = 'Type your sentence: ')
        self.title_label.pack()

        self.text_entry = tk.Entry(self.root)
        self.text_entry.pack()

        self.submit_button = tk.Button(self.root, text = 'Check sentiment', command = self.display_sentiment)
        self.submit_button.pack()

        self.result_label = tk.Label(self.root)
        self.result_label.pack()

        self.root.mainloop()

    def display_sentiment(self):
        sentence = self.text_entry.get()
        self.text_entry.delete(0, 'end')

        predicted = predict_sentiment(self.model, sentence, App.SENTIMENTS, self.max_length, self.vocab)
        self.result_label['text'] = predicted