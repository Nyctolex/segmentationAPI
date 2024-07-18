# segmentationAPI

#### Setup Environment 
- Create a virtual env using conda/python.
- Run `pip install -r requiement.txt` to install all dependencies.
- Notice that the version of Pytorch requires a Unix distribution.

## Phase one

The script for "Phase One" of the assignment can be found in the main folder under `phaseone.py`. You can run the script from the shell and provide an external command, such as specifying the picture to be used for inference. For example:
```
python3 phaseOne.py --photo ~/Desktop/sheep.jpeg 
```
If no picture is specified, the system will use its own default image.
The script runs the entire pipeline process specified in the instruction document and returns a final image showing the results of each model and the metrics measured for each one, as seen in the following demo:

![Figure_1](https://github.com/user-attachments/assets/8a291a7c-b383-4e60-8b0a-8a16df6ea938)

## PhaseTwo

The **"SegmentationModelAI”** class can be found in `models/model_wrappers.py` . As this object will be used later in **PhaseThree** for inference, there is no special script for addressing this model via the CLI. However, there are many examples of how to use this SDK in the unit tests under the file `tests/test_model_wrappers.py`.

## PhaseThree

In order to run the main service, you could run the main.py script. However a preferred approach would be using uvicorn as in the following command:
```
uvicorn main:app --reload
``` 

Then, follow this link in your browser: [http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/) and use the **“/api/infer”** rout to interact with the API using FastAPI's automatic GUI.
You can upload any supported image file (currently, any image type supported by the PIL library) and send it to the server to get the resulting image:

<img width="1094" alt="Screenshot 2024-07-18 at 16 01 03" src="https://github.com/user-attachments/assets/97f3ca2b-aae2-4660-ab38-77e434bc80de">

<img width="1086" alt="Screenshot 2024-07-18 at 16 02 55" src="https://github.com/user-attachments/assets/e42962fd-f402-415f-9e76-2be38f638a09">


![lox](https://github.com/user-attachments/assets/4a527e54-e277-49e4-bba7-9470fa6e0d9b)



