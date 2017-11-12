# CS 4242 - Game AI Project
## Objective
Our team seeks to develop a AI agent that is able to play a simple game. We plan to develop and train a learning model that will perform actions that help it to win the game. To accomplish this project, we plan to utilize libraries such as TensorFlow and OpenAI’s Gym in order to implement a way to train the agent to play the game proficiently.
## Milestones
 - [x] Decide on what game use for project. (CartPole)
 - [X] As a team, create different policies and see how they do.
 - [X] Create and train a learning model to play the game.
 - [X] Fine tune the model and optimize performance.
 - [X] Compare our initial policies with the learned model.
 - [ ] Discuss what we have learned.
 - [ ] Create PowerPoint and practice presentation.
 - [ ] Give presentation and submit project.
 - [ ] Receive A+ on the project.
 - [ ] :beer: or :coffee:

## Get Started
### Installing
#### Python
Install the latest Python 3 interpreter on your machine. Most Linux OS's should already have it. You can download Python 3 here:  <https://www.python.org/downloads/>

In order to execute a Python file, you can simply run the command:
```
python the_file_to_run.py
```

#### PIP
PIP is a popular package manager for Python. We'll use it to download gym, TensorFlow, etc. Follow these instructions to install it:  <https://pip.pypa.io/en/stable/installing/>

#### Other dependencies
To install all other dependencies, we will need to use PIP to install each package. We could type them in one by one, but I made a "requirements.txt" file which we can use to install all dependencies in bulk. In the same directory as the requirements file, run this in the command line:
```
pip install -r requirements.txt
```
**Note:** Please let me know if there are any issues. The only possible issue is with Linux, you may run into a problem installing `tkinter`. If so, please [follow the instructions here](https://stackoverflow.com/questions/4783810/install-tkinter-for-python).

### Getting the Project Code
If you're new to Git, no problem. If you would rather just play around with the project source code, just hit the download link to grab a `.zip`. **However**, I *really* recommend using Git, so that you can always update to the latest code, instead of re-downloading the .zip.

#### Installing Git
You can install Git here: <https://git-scm.com/downloads>

#### Cloning the Repo
Now, that you have Git installed, you can easily clone (or copy) the project source code. Simply, navigate to the directory you want to add the source code directory to and run the following:
```
git clone https://github.com/mtfuller/CS4242-project.git
```
Now, you should have a new folder in your current directory called `CS4242-project`.

**Note:** Windows users will need to use the "Git Bash" program that was installed during the Git installation to run Git commands.

### Play the CartPole game
If you haven't got to play it yet, try running the `play_game.py` file. This will launch a new CartPole game where you can move the cart with the left and right arrow keys.

### Create a new CartPole Policy
If you want to try your hand at creating a new policy for the CartPole, please take a look at `my_new_policy.py`. It is an example file that shows you how to create your own policy function, and then analyze it's performance. For example, `my_new_policy.py` creates two policies: a basic policy and a bad policy. After it uses the PolicyAnalyzer class to check the performance, the following bar graph is rendered to the screen:

![alt text](https://github.com/mtfuller/CS4242-project/raw/master/img/my_new_policy_graph.png "my_new_policy.py Graph")

You can see how much better the basic policy is than the bad policy. For more information, please see the well commented code in `my_new_policy.py`.

## Model Selection
As we were trying to come up with a agent policy, we soon saw the complexity of determining what action the agent should take, given four environment parameters. We stumbled across an example of developing a Q-Learning model to train an agent to play the CartPole game: <https://keon.io/deep-q-learning/>.

We began to train our model using the example and got adequate results. However, we found that the agent play the game very inconsistently, even when trained for thousands of runs.

## Optimizing the Model
The next step in the project was to fine tune the model, find the optimal values for gamma, learning rate, and other parameters. We created a script called `fine_tune_model.py` that would train multiple models, each with different values for each hyperparameter. After running the script, a few `.CSV` files would be generated. We evaluated the data and we found something significant when looking at the effect of changing the `gamma` value:

![alt text](https://github.com/mtfuller/CS4242-project/raw/master/img/optimized_DQL.png "Optimized DQL Graph")

We see that the `gamma` values and score means are a bit inversely proportional to each other. Meaning, the lower the `gamma`, the higher the average score is. Looking back over the performance data, one agent was able to play near perfect games, using a `gamma` of 0.9. Finally, we ran a performance test on four models we trained thus far:

![alt text](https://github.com/mtfuller/CS4242-project/raw/master/img/optimal_policies.png "Final Performance Test Graph")

We see that the optimized model is clearly able to play the game much more consistently (considering the standard deviation of 0) than earlier models.
