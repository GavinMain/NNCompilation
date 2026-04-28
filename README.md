# NNCompilation

If you have any feedback regarding the report or the code, please respond to this google form.
https://docs.google.com/forms/d/e/1FAIpQLSf74mQH2fEpyjPc5nkI0GByFDIBB1nN7yiA4N8QMK4RnvQF3w/viewform?usp=dialog

Any and all feedbcak are appreciated. 

To run the code, in the terminal run:
    chmod +x install.sh
    ./install.sh

Then navigate to any folder and run:
    python3 file.py

For the MLP and CNN, each file trains and evaluates a new model.
For the LLM and Diffusion models,
    download...py downloads datasets or pretrained models.
    _model.py defines the architecture as well as supplement structures like tokenizers and datasets.
    train.py trains the model. Some folders may have different training types such as SFT or RLHF.
    use_model.py runs the model, usually allowing the user to type in a prompt.
     