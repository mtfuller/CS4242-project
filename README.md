# CS 4242 - Game AI Project
## Objective
Our team seeks to develop a AI agent that is able to play a simple game. We plan to develop and train a learning model that will perform actions that help it to win the game. To accomplish this project, we plan to utilize libraries such as TensorFlow and OpenAI’s Gym in order to implement a way to train the agent to play the game proficiently.
## Milestones
 - [x] Decide on what game use for project. (CartPole)
 - [ ] As a team, create different policies and see how they do.
 - [ ] Create and train a learning model to play the game.
 - [ ] Fine tune the model and optimize performance.
 - [ ] Compare our initial policies with the learned model.
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

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

You can see how much better the basic policy is than the bad policy. For more information, please see the well commented code in `my_new_policy.py`.
