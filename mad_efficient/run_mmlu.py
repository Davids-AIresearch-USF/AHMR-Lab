import sys
import os
import argparse
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mad_framework import team
from datasets import load_dataset
import json



argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model",
    type=str,
    default="meta-llama/Llama-3.2-1B-Instruct",
    help="Model name. Can be any ollama model.",
)

argparser.add_argument(
    "--run",
    type=int,
    default=1,
    help="Number of tests.",
)

argparser.add_argument(
    "--num_agents",
    type=int,
    default=3,
    help="Number of agents in the debate.",
)

argparser.add_argument(
    "--round",
    type=int,
    default=1,
    help="Number of rounds in discussion.",
)


args = argparser.parse_args()
print(args)


model_filename = args.model
if "/" or ":" in args.model:
    model_filename = args.model.replace("/","_")


random_seed = 42
dataset = load_dataset("cais/mmlu", "all")

sample_size = 100
df = pd.DataFrame(dataset['test'])
df = df.sample(n=sample_size, random_state=random_seed)



system_text = f"""Engage in an active debate to determine the correct answer. Contribute your reasoning in no more than five sentences. \
Finalize your answer with the choice you believe is most appropriate (A, B, C, or D) and provide your confidence level \
(Confidence scale: 0% = very low confidence, 100% = very high confidence). Provide your decision in the following format:\n\
Answer: <A/B/C/D>; Confidence: <NN%>"""


mad_strategy = "standard"

for i in range(args.run):

    folder = f"results/mmlu_{mad_strategy}_{model_filename}_run_{i}_agents_{args.num_agents}"
    os.makedirs(folder, exist_ok=True)

    for j in range(len(df)):

        sample = df.iloc[j]
        question = sample['question']
        subject = sample['subject']
        choices = sample['choices']
        answer = sample['answer']

        letters = ["A", "B", "C", "D"]
        choices_formatted = "\n".join(f"{letters[i]}: {choice}" for i, choice in enumerate(choices))
        correct_answer = letters[answer]

        subject = subject.replace("_"," ")

        prompting_method = "Let's think step by step."
        user_text = f"""Subject: {subject}\nQuestion: {question}\nChoices:\n{choices_formatted}\n{prompting_method}"""


        # DEFINE AGENTS:
        participants = []
        for k in range(args.num_agents):
            participants.append(
                team.Agent(
                    name=f"Agent {k+1}",
                    persona="",
                    beliefs=[],
                    model=args.model
                )
            )


        # START THE DISCUSSION:
        speaking_pattern = list(range(args.num_agents))
        
        team_instance = team.Team(participants, speaking_pattern, strategy=mad_strategy)

        discussion_file = f"{folder}/mmlu_{mad_strategy}_{model_filename}_run_{i}_agents_{args.num_agents}_discussion_log_{j}.json"
        # Runs the debate
        discussion_log, belief_change_log = team_instance.kickoff(system_text, user_text, rounds=args.round)
        
        with open(discussion_file, "w") as file:
            json.dump(discussion_log, file, indent=4)








