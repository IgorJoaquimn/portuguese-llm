import openai
from promptl_ai import Promptl

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

# Initialize Promptl
promptl = Promptl()

# Define the PromptL template
prompt_template = """
<step>
    First, think step by step about how to answer the user's question.
    <user>
        Taking into account this context: {{context}}
        I have the following question: {{question}}
    </user>
</step>
<step>
    Finally, answer the user's question succinctly yet completely.
</step>
"""

# Function to generate a response using OpenAI API
def generate_response(context, question):
    # Format the prompt using Promptl
    formatted_prompt = promptl.chains.create(
        prompt=prompt_template,
        parameters={"context": context, "question": question}
    ).step()  # Get the formatted prompt

    # Send the formatted prompt to OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt.messages[-1]['content']}
        ],
        temperature=0.7
    )
    
    return response["choices"][0]["message"]["content"]

# Example usage
context = "PromptL is a templating language specifically designed for LLM prompting."
question = "What is PromptL?"

answer = generate_response(context, question)
print(answer)
