# ML-flask

This is the ML-flask repository for the Rahayoo Bangkit Capstone 2023 project. The repository contains the code for a Flask application that performs text classification using a pre-trained Bidirectional LSTM model.

## Installation

To install and run the ML-flask application, follow these steps:

1. Make sure you have Python installed on your system.
2. Clone this repository to your local machine: `git clone https://github.com/rahayoo-bangkit-capstone-2023/ML-flask.git`.
3. Open a terminal and navigate to the repository directory: `cd ML-flask`.
4. Install the required dependencies by running: `pip install -r requirements.txt`.
5. Download the pre-trained model file `Bidirectional LSTM English EMO.h5` and place it in the repository directory.
6. Start the Flask application by running: `python app.py`.

## Usage

Once the Flask application is running, you can use the following endpoint to perform text classification:

- `/api/deteksi` (POST): Performs text classification on the input text and returns the predicted class.

Example API request:

```bash
curl -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "data=Some text to classify" http://localhost:8080/api/deteksi
```

Response:

```json
{
  "data": "predicted_class"
}
```

## Preprocessing

The input text goes through several preprocessing steps before classification, including normalization, tokenization, and padding. The preprocessing steps are performed in the `preprocess()` function in the `app.py` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

If you have any questions or would like to contribute to this project, please contact the contributors:

- Rizky Pramudita
- Fanisa Nimastiti

Please reach out to them for more information about the ML-flask project.
