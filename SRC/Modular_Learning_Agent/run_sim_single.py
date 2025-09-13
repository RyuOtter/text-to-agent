from .utils_tools import get_tools
from .sgd_data_utils import prepare_goals_str
from .multiwoz_data_utils import prepare_goals_string
from .utils_llm import LLMModel
from .utils_sim import Agent

# Function used to simulate a single user-assistant conversation
def single_simulation(benchmark, individual_dialog, user, assistant, max_iterations=10, schemas=None, ground_truth_included = False):

    # Extract the task goal
    if benchmark == "SGD":
        task_goal = prepare_goals_str(individual_dialog)
    elif benchmark == "MultiWOZ":
        if isinstance(individual_dialog, dict):
            task_goal = prepare_goals_string(individual_dialog["goal"]["message"])
        else:
            raise ValueError("Single dialog expected, but multiple received")
    else:
        raise ValueError("Wrong benchmark")
    

    # Initialize agents
    user.include_user_goal(task_goal)

    if benchmark == "SGD":
        tools = get_tools(benchmark=benchmark, datapoint=individual_dialog, schemas=schemas, dynamic=True)
        assistant.set_tools(tools, schemas=schemas)

    # Loop throught the turns in a conversation
    simulated_conversation = []
    turn_idx = 1

    for iteration in range(max_iterations):

        # User message
        if ground_truth_included:
            user_response = user(simulated_conversation, individual_dialog)
        else:
            user_response = user(simulated_conversation)

        # Check for early stopping
        if "Task Completed" in user_response:
            simulated_conversation.append({
                "turn_idx": turn_idx,
                "user": user_response,
                "agent": "Goodbye."})
            break

        simulated_conversation.append({"turn_idx": turn_idx, "user": user_response})

        # Agent response
        assistant_response = assistant(simulated_conversation)

        simulated_conversation[-1]["agent"] = assistant_response
        turn_idx +=1
    
    # Extract tool logs
    function_logs = assistant.tool_log
    assistant.tool_log = []
    
    # Return simulation results
    if benchmark == "SGD":
        return simulated_conversation, function_logs
        
    if benchmark == "MultiWOZ":
        goal = individual_dialog.get("goal", {})
        goal_messages = individual_dialog.get("goal", {}).get("message", [])
        dialog_refer = individual_dialog.get("log", [])
        return simulated_conversation, function_logs, goal, goal_messages, dialog_refer