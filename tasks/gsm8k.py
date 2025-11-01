"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
from datasets import load_dataset
from tasks.common import Task

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], f"GSM8K subset must be main|socratic"
        assert split in ["train", "test"], f"GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2] # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts}
        ]

        conversation = {
            "messages": messages
        }

        return conversation


    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth
        assistant_message = conversation['messages'][-1]
        assert assistant_message["role"] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(ref_num == pred_num)
        return is_correct
    
    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex. (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
    
"""

python -m tasks.common
Length of MMLU: 7473

5th example is: {'messages': [{'role': 'user', 'content': "James decides to build a tin house by collecting 500 tins in a week. On the first day, he collects 50 tins. On the second day, he manages to collect 3 times that number. On the third day, he collects 50 tins fewer than the number he collected on the second day. If he collects an equal number of tins on the remaining days of the week, what's the number of tins he collected each day for the rest of the week?"}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'On the second day, he collected 3 times the number of tins he collected on the first day, which is 3*50 = '}, {'type': 'python', 'text': '3*50'}, {'type': 'python_output', 'text': '150'}, {'type': 'text', 'text': '150 tins.\nOn the third day, he collected 50 tins fewer than the second day, which is 150-50 = '}, {'type': 'python', 'text': '150-50'}, {'type': 'python_output', 'text': '100'}, {'type': 'text', 'text': '100 tins\nThe total for the three days is 150+100+50 = '}, {'type': 'python', 'text': '150+100+50'}, {'type': 'python_output', 'text': '300'}, {'type': 'text', 'text': '300 tins.\nTo reach his goal, he still needs 500-300 = '}, {'type': 'python', 'text': '500-300'}, {'type': 'python_output', 'text': '200'}, {'type': 'text', 'text': "200 tins.\nSince the total number of days left in the week is 4, he'll need to collect 200/4 = "}, {'type': 'python', 'text': '200/4'}, {'type': 'python_output', 'text': '50'}, {'type': 'text', 'text': '50 tins per day to reach his goal\n#### 50'}]}]}

Length of sliced MMLU[5:10]:  5

0th example of sliced MMLU:  {'messages': [{'role': 'user', 'content': "James decides to build a tin house by collecting 500 tins in a week. On the first day, he collects 50 tins. On the second day, he manages to collect 3 times that number. On the third day, he collects 50 tins fewer than the number he collected on the second day. If he collects an equal number of tins on the remaining days of the week, what's the number of tins he collected each day for the rest of the week?"}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'On the second day, he collected 3 times the number of tins he collected on the first day, which is 3*50 = '}, {'type': 'python', 'text': '3*50'}, {'type': 'python_output', 'text': '150'}, {'type': 'text', 'text': '150 tins.\nOn the third day, he collected 50 tins fewer than the second day, which is 150-50 = '}, {'type': 'python', 'text': '150-50'}, {'type': 'python_output', 'text': '100'}, {'type': 'text', 'text': '100 tins\nThe total for the three days is 150+100+50 = '}, {'type': 'python', 'text': '150+100+50'}, {'type': 'python_output', 'text': '300'}, {'type': 'text', 'text': '300 tins.\nTo reach his goal, he still needs 500-300 = '}, {'type': 'python', 'text': '500-300'}, {'type': 'python_output', 'text': '200'}, {'type': 'text', 'text': "200 tins.\nSince the total number of days left in the week is 4, he'll need to collect 200/4 = "}, {'type': 'python', 'text': '200/4'}, {'type': 'python_output', 'text': '50'}, {'type': 'text', 'text': '50 tins per day to reach his goal\n#### 50'}]}]}

They match:  True

10th Example of the sliced MMLU for viz:  {'messages': [{'role': 'user', 'content': 'Trace has five shopping bags that weigh the same amount as Gordon’s two shopping bags. One of Gordon’s shopping bags weighs three pounds and the other weighs seven. Trace’s shopping bags all weigh the same amount. How many pounds does one of Trace’s bags weigh?'}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Gordon’s bags weigh 3 + 7 = '}, {'type': 'python', 'text': '3+7'}, {'type': 'python_output', 'text': '10'}, {'type': 'text', 'text': '10 pounds.\nTrace’s five bags all weigh the same amount, so each bag weighs 10 / 5 = '}, {'type': 'python', 'text': '10/5'}, {'type': 'python_output', 'text': '2'}, {'type': 'text', 'text': '2 pounds.\n#### 2'}]}]}

"""