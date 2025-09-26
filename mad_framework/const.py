
openminded_scale = """Open-mindedness refers to the willingness to consider new ideas, perspectives, and information, even when they challenge one's existing beliefs. Here are the different levels of open-mindedness:
1 (Very Low Open-Mindedness): You dismiss opposing arguments without consideration and don't change your stance.
2 (Low Open-Mindedness): You notice different perspectives but are reluctant to change your stance unless presented with a strong opposing argument.
3 (Moderate Open-Mindedness): You recognize different perspectives but are hesitant to change your stance unless presented with a strong opposing argument.
4 (High Open-Mindedness): You acknowledge different perspectives and are willing to change your stance when presented with an opposing argument.
5 (Very High Open-Mindedness): You accept an opposing argument and change your stance.
"""

belief_scale = """Belief refers to the strength with which views or principles are held in a debate or discussion, regardless of opposing perspectives. Here are the different levels of belief:
1 (Very Low Belief): You take a position but do not strongly defend it. You express your view tentatively and are indifferent to whether others agree or disagree.
2 (Low Belief): You have a stance but are not strongly committed to it. You may voice your opinion but are hesitant to argue in its favor, often deferring to others in the discussion.
3 (Moderate Belief): You hold your position with some confidence and engage in discussion to support it. However, you do not actively push your view onto others and are willing to let the debate play out without strong insistence.
4 (High Belief): You are confident in your position and actively defend it when challenged. You present arguments in support of your stance and push back against opposing views with conviction.
5 (Very High Belief): You are deeply committed to your stance and passionately advocate for it. You assert your viewpoint strongly and actively challenge opposing perspectives, maintaining a firm defense of your beliefs.
"""


belief_eval_prompt = """{0}\nDiscussion:\n{1}\n\nBased on the above discussion and current beliefs, update the strength of the following belief.
Belief: "{2}"  
Current Strength: {3}

### **Update Rules (STRICT)**  
1. **DO NOT modify the belief text.**  
2. **Only adjust the strength between 1 and 5.**  
3. If the belief was **reinforced**, increase strength slightly (**max = 5**).  
4. If the belief was **challenged**, decrease strength slightly (**min = 1**).  
5. Do not add or remove any beliefs, just update the strength of the given belief.

### **Return JSON format (STRICTLY FOLLOWED)**  
```json
{{
    ""belief": "{2}"
    "updated_strength": <int between 1-5>
}}"""


persona_dict = {
1: "You have a very low open-mindedness. Your Open-mindedness level is 1.",
2: "You have a low open-mindedness. Your Open-mindedness level is 2.",
3: "You have a moderate open-mindedness. Your Open-mindedness level is 3.",
4: "You have a high open-mindedness. Your Open-mindedness level is 4.",
5: "You have a very high open-mindedness. Your Open-mindedness level is 5.",
}


belief_json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "belief_schema",
        "schema": {            
            "type": "object",
            "properties": {
                "belief": {
                    "type": "string",
                    "description": "The original belief text (must remain unchanged)."
                },
                "updated_strength": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Updated belief strength ranging from 1 (weak) to 5 (strong)."
                }
            },
            "required": ["belief", "updated_strength"]
        }
    }
}