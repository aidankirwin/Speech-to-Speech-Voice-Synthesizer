# Speech-to-Speech Voice Synthesizer

## Build Instructions (Mac)
1. Clone repository
2. Install conda (recommended: Anaconda distribution installer)
3. Open repository and create conda environment
```conda create --name synth python=3.10```
4. Activate environment
```conda activate synth```
5. Install modules
```conda install --file requirements.txt```
6. Check that python path is correct
```which python```
If this does not say ```/Users/[Your Username]/anaconda3/envs/synth/bin/python``` or something similar then restart your terminal and try the following:
- ```conda deactivate```
- ```conda activate synth```
- ```which python```
7. Run the program
```python voice_synthesizer.py```