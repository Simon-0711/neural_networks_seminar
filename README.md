# Neural Network Seminar
GitHub repository for the seminar project in "sequence 2 sequence learning neural networks" during the summer semester 2023.


# Model Exploration

#### Explored by Leon

##### Easy OCR
Tested on the test images located in `model_exploration/test_images`. Performance on whole document images reasonable, since the model already includes preprocessing and other architecture components. Test on food nutritin tables decent for a OCR model not specifically trained for this task. When running this make shure your have configured your data path inside `model_exploration/easy_ocr/config.yml`. Also the `tools.py` file has been extended by some methods to check the completeness of the collected data from Open Food Facts.

##### Microsoft Transformer OCR
Only testet the pretrained model on a test image located inside the directory `model_exploration/test_images`.
Extraction on whole document extremely random. Needs single lines of text as Input.