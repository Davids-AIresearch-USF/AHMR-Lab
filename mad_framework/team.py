from typing import List, Tuple
import json
import re
from pydantic import BaseModel, Field
from .const import belief_scale, belief_eval_prompt, belief_json_schema
from .huggingface_lib import load_model, get_output, get_belief_output
from copy import deepcopy
# import ollama
# from ollama_lib import get_output
# from openai_lib import get_gpt_output



class Belief(BaseModel):
    belief: str = Field(..., description="The original belief text (must remain unchanged).")
    updated_strength: int = Field(..., ge=1, le=5, description="Updated belief strength ranging from 1 (weak) to 5 (strong).")

class BeliefList(BaseModel):
    beliefs: List[Belief] = Field(..., description="the full set of beliefs. Number of beliefs must stay the same as before")  # List of belief objects
    # strengths: List[int] = Field(..., description="Updated list of belief strengths. Must match the number of original beliefs, with values between 1 (weak) and 5 (strength).")


class Agent:
    def __init__(self, name:str, persona:str, beliefs:List[Tuple[str, float]], model:str):
        """
        Initializes the Agent with the specified persona, task, and model.

        :param name: The name of the agent.
        :param persona: Background information or context for the agent.
        :param task: The goal the agent is tasked with achieving.
        :param model: The model used by the agent.
        """
        self.name:str = name
        self.persona:str = persona
        self.model:str = model
        self.beliefs:List[Tuple[str, float]] = beliefs
        self.tokenizer_load, self.model_load = load_model(self.model)

    def describe(self):
        """
        Prints the agent's details.
        """
        description = (
            f"Name: {self.name}\n"
            f"Persona: {self.persona}\n"
            f"Model: {self.model}\n"
            f"beliefs: {self.beliefs}\n"
        )


    def respond(self, system_text:str, task_prompt:str, agent_log:dict, message_log:dict):
        """
        Sends a message to the Ollama model and returns the response.
        :param message: The input message for the agent.
        :return: The response from the model.
        """
        belief = self.format_belief_box(self.beliefs)

        system_prompt = f"You are {self.name}. {system_text}"
        # system_prompt = f"{system_text}\n{belief_scale}\nYou are {self.name}. You will defend your beliefs in the discussion and present arguments to support them." #: {belief}"

        messages = [{"role": "system", "content": system_prompt}]

        curr_agent = self.name
        # number of rounds finished by the current agent so far:
        curr_agent_rounds = len(agent_log[curr_agent])

        if '1' in curr_agent:
            messages.append({"role": "user", "content": task_prompt})
            # messages.append({"role": "user", "content": f"Your Beliefs: {belief}\n" + task_prompt})
        # loop to add responses of all agents from previous rounds
        user_msgs = []
        for r in range(curr_agent_rounds):
            for agent in agent_log:
                if agent == curr_agent:
                    if len(user_msgs) > 0:
                        if messages[-1]["role"].lower() == "user":
                            messages[-1]["content"] = messages[-1]["content"] + "\n" + "\n".join(user_msgs) + "\n\n" + f"###\n\n" + task_prompt
                            # messages[-1]["content"] = messages[-1]["content"] + "\n" + "\n".join(user_msgs) + "\n\n" + f"###\n\nYour Beliefs: {belief}\n" + task_prompt
                        else:
                            messages.append({"role": "user", "content": "\n".join(user_msgs) + f"###\n\n" + task_prompt})
                            # messages.append({"role": "user", "content": "\n".join(user_msgs) + f"###\n\nYour Beliefs: {belief}\n" + task_prompt})
                        user_msgs = []
                    assistant_msg = agent_log[curr_agent][r].lstrip(f"{curr_agent}: ")
                    messages.append({"role": "assistant", "content": assistant_msg})
                    continue
                if agent != curr_agent and len(agent_log[agent]) > r:
                    user_msgs.append(agent_log[agent][r])
        if len(user_msgs) > 0:
            if messages[-1]["role"].lower() == "user":
                messages[-1]["content"] = messages[-1]["content"] + "\n" + "\n".join(user_msgs)  + "\n\n" + f"###\n\n" + task_prompt
                # messages[-1]["content"] = messages[-1]["content"] + "\n" + "\n".join(user_msgs)  + "\n\n" + f"###\n\nYour Beliefs: {belief}\n" + task_prompt
            else:
                messages.append({"role": "user", "content": "\n".join(user_msgs)})
        
        new_user_prompts = []
        for agent in agent_log.keys():
            if agent != curr_agent and len(agent_log[agent]) > curr_agent_rounds:
                new_user_prompts.append(agent_log[agent][curr_agent_rounds])  # Only get new messages
        # If any agents already went in the current round, add those to the first user prompt along with the task_prompt, else just add task_prompt
        if len(new_user_prompts) > 0:
            if messages[-1]["role"].lower() == "assistant" or messages[-1]["role"].lower() == "system":
                messages.append({"role": "user", "content": "\n".join(new_user_prompts) + "\n\n" + f"###\n\n" + task_prompt})
                # messages.append({"role": "user", "content": "\n".join(new_user_prompts) + "\n\n" + f"###\n\nYour Beliefs: {belief}\n" + task_prompt})
            else:
                messages[-1]["content"] = messages[-1]["content"] + "\n" + "\n".join(new_user_prompts) + "\n" + f"###\n\n" + task_prompt
                # messages[-1]["content"] = messages[-1]["content"] + "\n" + "\n".join(new_user_prompts) + "\n" + f"###\n\nYour Beliefs: {belief}\n" + task_prompt
        else:
            if messages[-1]["role"].lower() == "user":
                if len(messages) != 2:
                    messages[-1]["content"] = messages[-1]["content"] + "\n" + f"###\n\n" + task_prompt
                    # messages[-1]["content"] = messages[-1]["content"] + "\n" + f"###\n\nYour Beliefs: {belief}\n" + task_prompt

        message_log[self.name] = messages 
        

        print(f"agent:{self.name}")
        print(messages)

        if self.model in ["gpt-4o-mini-2024-07-18"]:        
            model_respond = get_gpt_output(model=self.model, msg=messages)
        else:

            # model_respond = ollama.chat(model=self.model, messages=messages)
            # model_respond = model_respond['message']['content']
            model_respond, token_count = get_output(tokenizer=self.tokenizer_load, model=self.model_load, msg=messages)

        # chat_respond = f"{model_respond}"
        chat_respond = f"{self.name}: {model_respond}"

        return chat_respond, token_count

    def eval(self, discussion:List[str]):
        curr_beliefs = (self.beliefs).copy()
        text_discussion = "\n".join(discussion)

        for i in range(len(self.beliefs)):
            system_prompt = f"Persona: You are {self.name}. {self.persona}\nCurrent Beliefs: {self.beliefs}"
            user_prompt = belief_eval_prompt.format(belief_scale, text_discussion, self.beliefs[i][0], self.beliefs[i][1])


            messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]


            if self.model in ["gpt-4o-mini-2024-07-18"]:        
                response = get_gpt_output(model=self.model, msg=messages, response_format=belief_json_schema)
            else:

                response = get_belief_output(tokenizer=self.tokenizer_load, model=self.model_load, msg=messages)
                
                try:
                    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
                    response = match.group(0)
                except:
                    print(response)

            try:
                updated_belief = json.loads(response)
                self.beliefs[i] = (self.beliefs[i][0], updated_belief["updated_strength"])

            except:
                self.beliefs[i] = (self.beliefs[i][0], 0)

            # self.beliefs[i] = (updated_belief["belief"], updated_belief["updated_strength"])
            # self.beliefs[i] = (self.beliefs[i][0], updated_belief["updated_strength"])



        return f"{self.name}\n\nStarting Beliefs:\n{curr_beliefs}\n\nUpdated Beliefs:\n{self.beliefs}" 


    def format_belief_box(self,belief_box):
        """
        Converts a belief box into a readable sentence format.
        Args:
            belief_box (list): A list of lists, where each inner list contains a belief statement and its confidence score.
        Returns:
            str: A formatted string summarizing the beliefs.
        """
        formatted_beliefs = []
        
        for belief, confidence in belief_box:
            formatted_beliefs.append(
                f"{belief} ({confidence} out of 5)"
            )
        
        return " \n".join(formatted_beliefs)




class Team:
    def __init__(self, agents, pattern, strategy="efficient"):
        """
        Initializes the Team with a list of agents and a speaking pattern.

        :param agents: A list of Agent objects.
        :param pattern: A list representing the speaking order pattern.
        :param strategy: A list representing strategy. Can be any from [efficient, belief].
        """
        self.agents = agents
        self.pattern = pattern
        self.strategy = strategy

    def kickoff(self, system_text, task_prompt, rounds=3, eval_rate = 1):

        """
        Starts a discussion between agents based on the speaking pattern.

        :param topic: The initial topic of discussion.
        :param rounds: Number of discussion rounds.
        """
        order = generate_custom_order(self.pattern, rounds)
        discussion = []
        prompt = task_prompt
        agent_log = {}
        message_log = {}
        for agent in self.agents:
            # list of responses for each agent:
            agent_log[agent.name] = []
            message_log[agent.name] = []

        prev_agent = None
        last_processed_index = {agent: 0 for agent in agent_log.keys()}

        belief_changes = {}
        for agent in self.agents:
            belief_changes[agent.name] = {"Initial": deepcopy(agent.beliefs)}

        discussion_dict = {}
        for turn in range(len(order)):
            round_num = (turn // len(self.pattern)) + 1
            print(f"\n\nRound: {round_num}\n")
            agent = self.agents[order[turn]]
            response, token_count = agent.respond(system_text, prompt, agent_log, message_log)

            agent_log[agent.name].append(response)
            discussion.append(response)
            prev_agent = self.agents[order[turn]].name

            # --- Build JSON discussion round by round ---
            round_key = f"Round {round_num}"
            if round_key not in discussion_dict:
                discussion_dict[round_key] = {}

            clean_response = response.split(':', 1)[1].lstrip()
            discussion_dict[round_key][agent.name] = {"output": clean_response, "prompt_token": token_count["prompt_token"], "generated_token": token_count["generated_token"]}


            if ((turn + 1) % len(self.pattern)) == 0: # at the end of a round clear the agents reponses [0, 1, 2, 3]
                print(f"\nEnd of round: {turn// len(self.pattern) + 1}")
                print("\n\n\n")


            if self.strategy == "belief":
                if (((turn + 1) % (len(self.pattern)*eval_rate)) == 0) or (turn + 1 == len(order)):
                    for agent in self.agents:
                        agent_change = agent.eval(discussion)
                        belief_changes[agent.name][f"Round {(turn// len(self.pattern)) + 1}"] = deepcopy(agent.beliefs)

        return discussion_dict, belief_changes
    

def generate_custom_order(pattern, repetitions):
    """
    Generate a custom speaking order based on a given pattern and repetitions.

    :param pattern: A list representing the repeating pattern (e.g., [0, 1, 2]).
    :param repetitions: The number of times to repeat the pattern.
    :return: A list representing the speaking order.
    """
    return pattern * repetitions


# Example Usage

def main():
    # Define agents
    agent1 = Agent(
        name="Agent 1",
        persona="A compassionate and intuitive therapist",
        beliefs = [
        ("Emotional well-being is as important as physical health", 4),
    ],
        model="meta-llama/Llama-3.2-1B-Instruct"  # "gpt-4o-mini-2024-07-18"
    )

    agent2 = Agent(
        name="Agent 2",
        persona="assertive, provocative, entrepreneurial, controversial, and self-confident influencer",
        beliefs = [
        ("Success is a result of hard work and strategic risk-taking", 4),
    ],
        model="meta-llama/Llama-3.2-1B-Instruct"  # "gpt-4o-mini-2024-07-18"
    )

    agent3 = Agent(
        name="Agent 3",
        persona="assertive, provocative, entrepreneurial, controversial, and self-confident influencer",
        beliefs = [
        ("Success is a result of hard work and strategic risk-taking", 4),
    ],
        model="meta-llama/Llama-3.2-1B-Instruct"  # "gpt-4o-mini-2024-07-18"
    )

    agent4 = Agent(
        name="Agent 4",
        persona="assertive, provocative, entrepreneurial, controversial, and self-confident influencer",
        beliefs = [
        ("Success is a result of hard work and strategic risk-taking", 4),
    ],
        model="meta-llama/Llama-3.2-1B-Instruct"  # "gpt-4o-mini-2024-07-18"
    )


    speaking_pattern = [0, 1, 2, 3]
    mad_strategy = "efficient"

    # Start the discussion
    participants = [agent1, agent2, agent3, agent4]
    team = Team(participants, speaking_pattern, strategy=mad_strategy)
    discussion_topic = "Does the new generation need college education?"
    # updates belief box at the end of round 3 (custom input)
    discussion_log, belief_change_log = team.kickoff(discussion_topic, discussion_topic, rounds = 2, eval_rate = 1)

    print(discussion_log)

    discussion_file = "discussion_log.json"
    with open(discussion_file, "w") as file:
        json.dump(discussion_log, file, indent=4)


    if mad_strategy == "belief":
        belief_change_file = "belief_change_log.json"
        with open(belief_change_file, "w") as file:
            json.dump(belief_change_log, file, indent=4)
        print(belief_change_log)



if __name__ == "__main__":
    main()

