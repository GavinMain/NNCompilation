# NNCompilation

If you have any feedback regarding the report or the code, please respond to this Google Form:  
https://docs.google.com/forms/d/e/1FAIpQLSf74mQH2fEpyjPc5nkI0GByFDIBB1nN7yiA4N8QMK4RnvQF3w/viewform?usp=dialog

Any and all feedback is appreciated.

---

## Running the Code

In the terminal, run:

```bash
chmod +x install.sh
./install.sh
```
## Project Structure

### MLP and CNN
- Each file trains and evaluates a new model.

### LLM and Diffusion Models
- `download...py`  
  Downloads datasets or pretrained models.

- `_model.py`  
  Defines the architecture as well as supporting structures like tokenizers and datasets.

- `train.py`  
  Trains the model. Some folders may include different training types such as SFT or RLHF.

- `use_model.py`  
  Runs the model, usually allowing the user to input a prompt.