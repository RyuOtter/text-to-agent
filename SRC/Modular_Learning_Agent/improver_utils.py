
import json

# Summarizer agent
def summarizer(simulated_conversations, tool_logs, summarizer_llm):

    system_prompt = """
    YOUR TASK:
    You are an expert in analyzing AI assistant's behavior in user facing conversations.
    You will be given a conversation between a user and an AI assistant and the logs of the tools used by the assistant.
    Your goal is to identify shortcomings of the assistant in the conversation, collect examples of them and score their severity.

    HOW TO SOLVE YOUR TASK:
    Please follow these 3 steps to solve you task closely:

    Step 1: Analyze the conversation and tool logs
    - Carefully read the assistant's responses and the tool logs for each conversation
    - CRITICAL: Cross-reference tool logs with assistant responses to identify mismatches between what tools returned vs what assistant claimed
    - Look for shortcomings of all nature
    - Pay particular attention to the following four categories:
        - Task progression: Clarification questions, getting stuck in loops, understanding of user intents
        - Conversational style: Natural flow, human-likeness, conciseness, no role confusion
        - Tool usage: Incorrect tool usage, leakage of json / system formats into conversation, tool result misrepresentation
        - Tool-response grounding: Compare tool logs with assistant responses to catch hallucination, fabrication, or misattribution of tool results
    - Collect ALL the issues that the assistant has in the conversations
    - Make sure that you only focus on assistant shortcoming and not user errors

    Step 2: Severity scoring
   - Range from 0 (low severity) to 1 (high severity)
   - Based on frequency within one conversation
   - Based on occurrence across all dialogues of the batch
   - Based on impact on user goal and conversation

    Step 3: Issue selection and formulation
    - Choose the top 3 most severe issues from your analysis
    - You can also choose less than 3 issues if you think the reamining issues are not severe enough
    - Formulate the issue in a concise and clear manner

    Step 4: Example extraction
    - Extract a highly representative example of the issue that clearly portrays the issue
    - Keep the example short and concise

    OUTPUT FORMAT:
    Strictly output a JSON object with the following fields and no preceeding or postceeding text:
    {
    "issue_1": {
        "what": "Assistant ends the task after planning but never executes the final step",
        "example": "Okay, I will stop here.",
        "severity": 0.6},
    "issue_2": {
        "what": "Assistant ignores LookupAPI even when SKU is present",
        "example": "I'll just assume the price is £29.99…",
        "severity":0.9}
    }
    """

    all_conversations_text = ""
    for i, conv in enumerate(simulated_conversations):
        all_conversations_text += f"Conversation {i+1}:\n{str(conv)}\n\n"
    
    all_tool_logs_text = ""
    for i, logs in enumerate(tool_logs):
        all_tool_logs_text += f"Tool logs {i+1}:\n{str(logs)}\n\n"

    user_prompt = f"""
    Please analyze the following conversations and tool logs and output the JSON summary:
    
    Conversations:
    {all_conversations_text}

    Tool logs:
    {all_tool_logs_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    json_summary = summarizer_llm.chat(messages)

    return json_summary

# Improver agent
def improver(method, assistant_prompt, summary, improver_llm, summarizer_llm):
    if method == "vanilla":
        return vanilla_improver(assistant_prompt, summary, improver_llm)
    
    elif method == "CoT":
        return CoT_improver(assistant_prompt, summary, improver_llm)
    
    elif method == "self_consistency":
        CoT_responses = self_consistency_improver(assistant_prompt, summary, improver_llm)
        new_system_prompt = aggregate_CoT_responses(CoT_responses, assistant_prompt, summary, summarizer_llm)
        return new_system_prompt
    
    elif method == "self_refine":
        initial_system_prompt = vanilla_improver(assistant_prompt, summary, improver_llm)
        return self_refine_improver(initial_system_prompt, assistant_prompt, summary, improver_llm, summarizer_llm)
    
    else:
        raise ValueError("Invalid improver method")

# Vanilla improver
def vanilla_improver(assistant_prompt, summary, improver_llm):
    
    system_prompt = """
    YOUR TASK:
    You are an expert in improving system prompts from AI assistants.
    Your goal is to revise the current system prompt according to identified issues from conversations where it has been used.
    You will be given a summary of issues from past conversations and the current system prompt.

    HOW TO SOLVE YOUR TASK
    Please follow these steps to solve you task closely:

    Step 1: Review issues
    - Carefully read the summary of the issues
    - Try to fully understand the issues by examining the given examples and the severity score

    Step 2: Reread system prompt mapping issues to the prompt
    - Read the current system prompt. Try to understand how it is structured and functions
    - Identify which current components and which missing components lead to the issues
    - Pay particular attention to the fact that missing components can have a significant impact on the assistant's behavior
    - Pay particular attention to the fact that contradicting and unclear components can worsen the assistant's behavior

    Step 3: Restructure the entire system prompt
    - Based on your mapping of issues to the system prompt, decide whether the system prompt requires a structural change
    - Typically, this is one of the most effective changes to make and is helpful for high severity issues. Be open to larger structural changes. Think about how the ideal system prompt for this setting and the given issues would look like.
    - Pay particular attention to uncovered areas and missing sections.
    - Use best practices from prompt engineering such as step by step instrucitons, rules, examples, role clarification, style guidance, crtierias, few shot examples, evaluation policies, etc.
    - IMPORTANT: Never remove or restructure the core output format structure or input placeholders

    Step 4: Tune components
    - Based on your mapping of issues to the system prompt, decide whether smaller components of the system prompt require tuning
    - If yes, tune components that are most relevant to the issues
    - Be mindful of efficient writing of system prompts. Avoid redundancy, overly long bullet lists and overengineering.
    - You CAN improve tool usage instructions if issues are related to tool calling behavior
    - IMPORTANT: Never tune the basic JSON output format structure or input placeholders

    Step 5: Revision of the new system prompt
    - Ensure that the new system prompt is an improvement over the old one and addresses the issues
    - All you changes MUST be grounded in the issues identified. Do not make up changes that are not related and not grounded in the issues.
    - Instead of just adding bullets, think about whether a new structure or section might be more effective
    - Ensure that the new system prompt is not artificially blown up with too many components and avoid redundancy. Summarize given components of the system prompt concisely to improve clearness of command for the AI assistant
    
    PROMPT LENGTH MANAGEMENT:
    - If the current prompt is already long (>80 lines), prioritize consolidation over addition
    - Remove redundant or overlapping instructions before adding new ones
    - Merge similar rules into concise, unified ones
    - Replace verbose explanations with clear, actionable statements
    - Only add new content if it directly addresses high-severity issues (>0.8)
    - For lower-severity issues, improve existing sections rather than adding new ones

    THE MOST IMPORTANT RULE:
    The following core elements cannot be removed and MUST always be present in the new system prompt:
    - Basic JSON tool calling format (preserve existing format, or add appropriate format if missing):
      - If MultiWOZ: For query tools: {"tool": "query_domain", "args": {"sql": "SELECT [columns] FROM [table] WHERE [conditions]"}}
                     For booking tools: {"tool": "book_domain", "args": {"param1": "value1", "param2": "value2"}}
      - If SGD: {"tool": "tool_name", "args": {"param1": "value1", "param2": "value2"}}
    - Input placeholders: {conversation} and {tool_list}
    - Core tool output requirement: "Output nothing except the JSON object when calling a tool"
    - CRITICAL: Do not modify existing tool call format - only preserve it or add if completely missing
    
    However, you CAN improve and enhance:
    - Tool selection guidance and strategy
    - Argument handling and validation rules
    - Tool result processing instructions
    - Specific tool usage patterns and best practices

    OUTPUT FORMAT:
    Output only the new system prompt without any explanations as pure text.
    """

    user_prompt = f"""
    Please revise the following system prompt according to the following issues:

    Current system prompt:
    {assistant_prompt}

    Summarized issues:
    {summary}

    Output only the new system prompt without any explanations as pure text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    new_system_prompt = improver_llm.chat(messages)

    return new_system_prompt

# CoT improver
def CoT_improver(assistant_prompt, summary, improver_llm):
    
    system_prompt = """
    YOUR TASK:
    You are an expert in improving system prompts from AI assistants.
    Your goal is to revise the current system prompt according to identified issues from conversations where it has been used.
    You will be given a summary of issues from past conversations and the current system prompt.

    HOW TO SOLVE YOUR TASK
    Please follow these steps to solve you task closely:

    Step 1: Review issues
    - Carefully read the summary of the issues
    - Try to fully understand the issues by examining the given examples and the severity score

    Step 2: Reread system prompt mapping issues to the prompt
    - Read the current system prompt. Try to understand how it is structured and functions
    - Identify which current components and which missing components lead to the issues
    - Pay particular attention to the fact that missing components can have a significant impact on the assistant's behavior
    - Pay particular attention to the fact that contradicting and unclear components can worsen the assistant's behavior

    Step 3: Restructure the entire system prompt
    - Based on your mapping of issues to the system prompt, decide whether the system prompt requires a structural change
    - Typically, this is one of the most effective changes to make and is helpful for high severity issues. Be open to larger structural changes. Think about how the ideal system prompt for this setting and the given issues would look like.
    - Pay particular attention to uncovered areas and missing sections.
    - Use best practices from prompt engineering such as step by step instrucitons, rules, examples, role clarification, style guidance, crtierias, few shot examples, evaluation policies, etc.
    - IMPORTANT: Never remove or restructure the core output format structure or input placeholders

    Step 4: Tune components
    - Based on your mapping of issues to the system prompt, decide whether smaller components of the system prompt require tuning
    - If yes, tune components that are most relevant to the issues
    - Be mindful of efficient writing of system prompts. Avoid redundancy, overly long bullet lists and overengineering.
    - You CAN improve tool usage instructions if issues are related to tool calling behavior
    - IMPORTANT: Never tune the basic JSON output format structure or input placeholders

    Step 5: Revision of the new system prompt
    - Ensure that the new system prompt is an improvement over the old one and addresses the issues
    - All you changes MUST be grounded in the issues identified. Do not make up changes that are not related and not grounded in the issues.
    - Instead of just adding bullets, think about whether a new structure or section might be more effective
    - Ensure that the new system prompt is not artificially blown up with too many components and avoid redundancy. Summarize given components of the system prompt concisely to improve clearness of command for the AI assistant
    
    PROMPT LENGTH MANAGEMENT:
    - If the current prompt is already long (>80 lines), prioritize consolidation over addition
    - Remove redundant or overlapping instructions before adding new ones
    - Merge similar rules into concise, unified ones
    - Replace verbose explanations with clear, actionable statements
    - Only add new content if it directly addresses high-severity issues (>0.8)
    - For lower-severity issues, improve existing sections rather than adding new ones

    THE MOST IMPORTANT RULE:
    The following core elements cannot be removed and MUST always be present in the new system prompt:
    - Basic JSON tool calling format (preserve existing format, or add appropriate format if missing):
      - If MultiWOZ: For query tools: {"tool": "query_domain", "args": {"sql": "SELECT [columns] FROM [table] WHERE [conditions]"}}
                     For booking tools: {"tool": "book_domain", "args": {"param1": "value1", "param2": "value2"}}
      - If SGD: {"tool": "tool_name", "args": {"param1": "value1", "param2": "value2"}}
    - Input placeholders: {conversation} and {tool_list}
    - Core tool output requirement: "Output nothing except the JSON object when calling a tool"
    - CRITICAL: Do not modify existing tool call format - only preserve it or add if completely missing
    
    However, you CAN improve and enhance:
    - Tool selection guidance and strategy
    - Argument handling and validation rules
    - Tool result processing instructions
    - Specific tool usage patterns and best practices

    OUTPUT FORMAT:
    Think step by step. First output your reasoning within the token <thought> ... </thought> followed by your final answer <answer> ... </answer>.
    Your output must follow the following format:
    <thought>
    ... Your internal thinking: your step by step reasoning process about the issues and improvements to the system prompt.
    </thought>
    <answer>
    ... Your final answer: the new system prompt without any explanations as pure text.
    </answer>
    """

    user_prompt = f"""
    Please revise the following system prompt according to the following issues:

    Current system prompt:
    {assistant_prompt}

    Summarized issues:
    {summary}

    Output your reasoning and the new system prompt.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    CoT_response = improver_llm.chat(messages)

    try:
        new_system_prompt = CoT_response.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError:
        print("<answer> tags not found in CoT response")
        new_system_prompt = CoT_response.strip()

    return new_system_prompt


# Self-consistency improver
def self_consistency_improver(assistant_prompt, summary, improver_llm):
    
    system_prompt = """
    YOUR TASK:
    You are an expert in improving system prompts from AI assistants.
    Your goal is to revise the current system prompt according to identified issues from conversations where it has been used.
    You will be given a summary of issues from past conversations and the current system prompt.

    HOW TO SOLVE YOUR TASK
    Please follow these steps to solve you task closely:

    Step 1: Review issues
    - Carefully read the summary of the issues
    - Try to fully understand the issues by examining the given examples and the severity score

    Step 2: Reread system prompt mapping issues to the prompt
    - Read the current system prompt. Try to understand how it is structured and functions
    - Identify which current components and which missing components lead to the issues
    - Pay particular attention to the fact that missing components can have a significant impact on the assistant's behavior
    - Pay particular attention to the fact that contradicting and unclear components can worsen the assistant's behavior

    Step 3: Restructure the entire system prompt
    - Based on your mapping of issues to the system prompt, decide whether the system prompt requires a structural change
    - Typically, this is one of the most effective changes to make and is helpful for high severity issues. Be open to larger structural changes. Think about how the ideal system prompt for this setting and the given issues would look like.
    - Pay particular attention to uncovered areas and missing sections.
    - Use best practices from prompt engineering such as step by step instrucitons, rules, examples, role clarification, style guidance, crtierias, few shot examples, evaluation policies, etc.
    - IMPORTANT: Never remove or restructure the core output format structure or input placeholders

    Step 4: Tune components
    - Based on your mapping of issues to the system prompt, decide whether smaller components of the system prompt require tuning
    - If yes, tune components that are most relevant to the issues
    - Be mindful of efficient writing of system prompts. Avoid redundancy, overly long bullet lists and overengineering.
    - You CAN improve tool usage instructions if issues are related to tool calling behavior
    - IMPORTANT: Never tune the basic JSON output format structure or input placeholders

    Step 5: Revision of the new system prompt
    - Ensure that the new system prompt is an improvement over the old one and addresses the issues
    - All you changes MUST be grounded in the issues identified. Do not make up changes that are not related and not grounded in the issues.
    - Instead of just adding bullets, think about whether a new structure or section might be more effective
    - Ensure that the new system prompt is not artificially blown up with too many components and avoid redundancy. Summarize given components of the system prompt concisely to improve clearness of command for the AI assistant
    
    PROMPT LENGTH MANAGEMENT:
    - If the current prompt is already long (>80 lines), prioritize consolidation over addition
    - Remove redundant or overlapping instructions before adding new ones
    - Merge similar rules into concise, unified ones
    - Replace verbose explanations with clear, actionable statements
    - Only add new content if it directly addresses high-severity issues (>0.8)
    - For lower-severity issues, improve existing sections rather than adding new ones

    THE MOST IMPORTANT RULE:
    The following core elements cannot be removed and MUST always be present in the new system prompt:
    - Basic JSON tool calling format (preserve existing format, or add appropriate format if missing):
      - If MultiWOZ: For query tools: {"tool": "query_domain", "args": {"sql": "SELECT [columns] FROM [table] WHERE [conditions]"}}
                     For booking tools: {"tool": "book_domain", "args": {"param1": "value1", "param2": "value2"}}
      - If SGD: {"tool": "tool_name", "args": {"param1": "value1", "param2": "value2"}}
    - Input placeholders: {conversation} and {tool_list}
    - Core tool output requirement: "Output nothing except the JSON object when calling a tool"
    - CRITICAL: Do not modify existing tool call format - only preserve it or add if completely missing
    
    However, you CAN improve and enhance:
    - Tool selection guidance and strategy
    - Argument handling and validation rules
    - Tool result processing instructions
    - Specific tool usage patterns and best practices

    OUTPUT FORMAT:
    Think step by step. First output your reasoning within the token <thought> ... </thought> followed by your final answer <answer> ... </answer>.
    Your output must follow the following format:
    <thought>
    ... Your internal thinking: your step by step reasoning process about the issues and improvements to the system prompt.
    </thought>
    <answer>
    ... Your final answer: the new system prompt without any explanations as pure text.
    </answer>
    """

    user_prompt = f"""
    Please revise the following system prompt according to the following issues:

    Current system prompt:
    {assistant_prompt}

    Summarized issues:
    {summary}

    Output your reasoning and the new system prompt.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    CoT_responses = []
    for iteration in range(10):
        try:
            CoT_response = improver_llm.chat(messages)
            CoT_responses.append(CoT_response)
        except Exception as e:
            print(f"CoT response {iteration+1} failed: {e}")

    if not CoT_responses:
        raise ValueError("All CoT responses failed")

    return CoT_responses

# Aggregate CoT responses into one final improvement for self-consistency
def aggregate_CoT_responses(CoT_responses, assistant_prompt, summary, summarizer_llm):

    system_prompt = """
    YOUR TASK:
    You are an expert in self-consistency for chain of thought prompting.
    You will be given a list of 10 system prompt improvements and your task is to aggregate them into a single new system prompt.
    You will also be given the current old system prompt, a summary of its issues and different reasoning paths for the 10 new system prompt suggestions.

    HOW TO SOLVE YOUR TASK:
    Please follow these steps carefully:

    Step 1: Review the old system prompt and its issues
    - Carefully read the old system prompt and the summary of its issues
    - Try to fully understand which structure and components of the old system prompt are causal for the issues

    Step 2: Review the 10 new system prompt suggestions and their reasoning paths
    - Carefully read the 10 new system prompt suggestions and their reasoning paths

    Step 3: Aggregate th 10 new system prompts into one final new suggestion
    - Do not try to include everything from all the 10 suggestions. Focus on the most impactful components and structures for improving the old system prompt.
    - Use the prompt structure and the components observed in the 10 suggested prompts. Choose either based on impactfulness or on which are the most common.
    - Aggregate your selected structure and components into one coherent new system prompt
    - Ensure that the new system prompt is an improvement over the old one

    IMPORTANT:
    - The final system prompt must contain the following core elements:
        - Basic JSON tool calling format: {"tool": "tool_name", "args": {...}}
        - Input placeholders: {conversation} and {tool_list}
        - Core tool output requirement: "Output nothing except the JSON object when calling a tool"
    
    OUTPUT FORMAT:
    Strictly output only the final new system prompt in plain text and not any preceeding or postceeding reasoning.
    """

    user_prompt = f"""
    Please aggregate the following CoT responses into a single new system prompt.
    Output only the new system prompt in plain text and not any preceeding or postceeding reasoning::
    CoT responses:
    {CoT_responses}

    Old system prompt:
    {assistant_prompt}

    Summary of issues:
    {summary}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    new_system_prompt = summarizer_llm.chat(messages)

    return new_system_prompt

# Self-refine improver
def self_refine_improver(initial_system_prompt, assistant_prompt, summary, improver_llm, summarizer_llm):

    for iteration in range (3):

        feedback_system_prompt = """
        YOUR TASK:
        You are an expert in self-refinement of prompts.
        Your goal is to provide feedback to a system prompt proposal, given the original system prompt and the summarized issues of the original system prompt.

        HOW TO SOLVE YOUR TASK:
        Please follow these steps to solve your task closely:

        Step 1: Review the original system prompt and its issues
        - Carefully read the original system prompt and the summary of its issues
        
        Step 2: Review the system prompt proposal
        - Carefully read the system prompt proposal
        - Provide feedback on the clarity and conciseness of the proposal
        - Examine the structure of the proposal. Is it most effective? Are there missing sections? Could the secitons be condensed?
        - Examine the components of the proposal. Are they most effective? Are there missing components? Could the components be condensed?
        - Compare the proposal to best practices in prompt engineering such as examples, rules, step by step process descriptions, etc.

        IMPORTANT:
        - Ground your feedback in the original system prompt and the summarized issues.
        - Do not make up feedback that is not related to the issues.
        - Never suggest to remove or restructure the core output format, input placeholders or tool calling format.

        OUTPUT FORMAT:
        Output your feedback in plain text.
        """

        feedback_user_prompt = f"""
        Provide feedback to the system prompt proposal:
        {initial_system_prompt}

        Given the original system prompt:
        {assistant_prompt}

        And given the summarized issues of the original system prompt:
        {summary}
        """

        messages = [
            {"role": "system", "content": feedback_system_prompt},
            {"role": "user", "content": feedback_user_prompt}
        ]
        feedback = improver_llm.chat(messages)

        refinement_system_prompt = """
        YOUR TASK:
        You are an expert in refining system prompts.
        You are given a system prompt proposal and feedback to it. Your goal is to refine the proposal based on the feedback to make it more effective.

        HOW TO SOLVE YOUR TASK:
        Please follow these steps to solve your task closely:

        Step 1: Review the feedback
        - Carefully read the feedback
        - Relate the feedback to the original proposal

        Step 2: Rewrite the system prompt
        - Based on the feedback, examine whether structural changes of the secitons are needed
        - Based on the feedback, examine whether components and bullets of the system prompt require ccahnges
        - Based on the feedback, analyse whether the language used for the system prompt requires changes to be more effective.
        - Based on the feedback, examine whether prompt engineer best practices need to be applied.

        IMPORTANT:
        - Never remove or restructure the core output format, input placeholders or tool calling format.

        OUTPUT FORMAT:
        Strictly only output the refined system prompt in plain text.
        """

        refinement_user_prompt = f"""
        Refine the following system prompt proposal based on the feedback below:

        System prompt proposal:
        {initial_system_prompt}

        Feedback:
        {feedback}
        """

        messages = [
            {"role": "system", "content": refinement_system_prompt},
            {"role": "user", "content": refinement_user_prompt}
        ]
        refined_prompt = improver_llm.chat(messages)
        initial_system_prompt = refined_prompt
    
    return initial_system_prompt

# Logging of changes in system prompt across iterations
def prompt_change_analysis(new_system_prompt, old_system_prompt, analysis_llm):

    system_prompt = """
    YOUR TASK:
    You are an expert in analyzing changes in system prompts.
    You will be given an old system prompt and a new system prompt.
    Your goal is to identify the changes that have been made to the system prompt.

    OUTPUT FORMAT:
    Strictly output a JSON object with the following fields:
    {
        "no_change": <boolean>,
        "changes": [
            "<strictly concise summary of change #1>",
            "<strictly concise summary of change #2>",
            ...
        ]
    }
    """

    user_prompt = f"""
    Please analyze what changed in the following old and new system prompts:
    Old system prompt:
    {old_system_prompt}

    New system prompt:
    {new_system_prompt}

    Strictly output a JSON object.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt_change_analysis = analysis_llm.chat(messages)

    return prompt_change_analysis
