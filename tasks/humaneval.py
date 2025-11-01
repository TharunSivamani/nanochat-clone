"""
Evaluate the Chat model on HumanEval dataset.
Btw this dataset is a misnomer and has nothing to do with humans.
It is a coding benchmark.
"""

import re
from datasets import load_dataset
from nanochat.execution import execute_code
from tasks.common import Task

def extract_imports(prompt):
    """Extract import statements from the beginning of a code block."""
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            # Stop at first non-import, non-comment line
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    Extract Python code from LLM completion.

    Handles various output formats:
    - Code wrapped in ```python ... ``` or ``` ... ``` blocks
    - Plain code without markdown blocks
    - Extra text before/after code blocks

    Returns the first code block if found, otherwise returns the whole completion.
    """
    # Try to find markdown code blocks (```python or just ```)
    # Match ```python\n...\n``` or ```\n...\n```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    # No code blocks found, return the whole completion
    return completion.strip()

class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__()
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'
    
    def num_examples(self):
        return len(self.ds)
    
    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        prompt = row['prompt'] # prompts in HumanEval are the beginning of the program
        solution = row['canonical_solution'] # the correct continuation of the program
        entry_point = row['entry_point'] # the function to check
        test = row['test'] # the test cases
        complete_solution = f"{prompt}\n{solution}" 
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point, # needed during evaluation
            "test": test, # needed during evaluation
        }
        return conversation
    
    def evaluate(self, conversation, completion):
        """ Given (conversation, completion), return boolean success of the completion. """
        # the prompt will contain the imports and the function signature
        imports = extract_imports(conversation['messages'][0]['content'])
        # the completion will usually contain the whole function
        # but not always with the needed imports, so we manually append them
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        success = result.success
        return success
    
"""

# Length of MMLU: 164

# 5th example is: {'messages': [{'role': 'user', 'content': '\ndef words_string(s):\n    """\n    You will be given a string of words separated by commas or spaces. Your task is\n    to split the string into words and return an array of the words.\n    \n    For example:\n    words_string("Hi, my name is John") == ["Hi", "my", "name", "is", "John"]\n    words_string("One, two, three, four, five, six") == ["One", "two", "three", "four", "five", "six"]\n    """\n'}, {'role': 'assistant', 'content': '\ndef words_string(s):\n    """\n    You will be given a string of words separated by commas or spaces. Your task is\n    to split the string into words and return an array of the words.\n    \n    For example:\n    words_string("Hi, my name is John") == ["Hi", "my", "name", "is", "John"]\n    words_string("One, two, three, four, five, six") == ["One", "two", "three", "four", "five", "six"]\n    """\n\n    if not s:\n        return []\n\n    s_list = []\n\n    for letter in s:\n        if letter == \',\':\n            s_list.append(\' \')\n        else:\n            s_list.append(letter)\n\n    s_list = "".join(s_list)\n    return s_list.split()\n'}], 'entry_point': 'words_string', 'test': 'def check(candidate):\n\n    # Check some simple cases\n    assert True, "This prints if this assert fails 1 (good for debugging!)"\n    assert candidate("Hi, my name is John") == ["Hi", "my", "name", "is", "John"]\n    assert candidate("One, two, three, four, five, six") == ["One", "two", "three", "four", "five", "six"]\n    assert candidate("Hi, my name") == ["Hi", "my", "name"]\n    assert candidate("One,, two, three, four, five, six,") == ["One", "two", "three", "four", "five", "six"]\n\n    # Check some edge cases that are easy to work out by hand.\n    assert True, "This prints if this assert fails 2 (also good for debugging!)"\n    assert candidate("") == []\n    assert candidate("ahmed     , gamal") == ["ahmed", "gamal"]\n\n'}

# Length of sliced MMLU[5:10]:  164

# 0th example of sliced MMLU:  {'messages': [{'role': 'user', 'content': '\n\ndef below_threshold(l: list, t: int):\n    """Return True if all numbers in the list l are below threshold t.\n    >>> below_threshold([1, 2, 4, 10], 100)\n    True\n    >>> below_threshold([1, 20, 4, 10], 5)\n    False\n    """\n'}, {'role': 'assistant', 'content': '\n\ndef below_threshold(l: list, t: int):\n    """Return True if all numbers in the list l are below threshold t.\n    >>> below_threshold([1, 2, 4, 10], 100)\n    True\n    >>> below_threshold([1, 20, 4, 10], 5)\n    False\n    """\n\n    for e in l:\n        if e >= t:\n            return False\n    return True\n'}], 'entry_point': 'below_threshold', 'test': '\n\nMETADATA = {}\n\n\ndef check(candidate):\n    assert candidate([1, 2, 4, 10], 100)\n    assert not candidate([1, 20, 4, 10], 5)\n    assert candidate([1, 20, 4, 10], 21)\n    assert candidate([1, 20, 4, 10], 22)\n    assert candidate([1, 8, 4, 10], 11)\n    assert not candidate([1, 8, 4, 10], 10)\n\n'}

# They match:  False

# 10th Example of the sliced MMLU for viz:  {'messages': [{'role': 'user', 'content': '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)\n\n\ndef decode_cyclic(s: str):\n    """\n    takes as input string encoded with encode_cyclic function. Returns decoded string.\n    """\n'}, {'role': 'assistant', 'content': '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)\n\n\ndef decode_cyclic(s: str):\n    """\n    takes as input string encoded with encode_cyclic function. Returns decoded string.\n    """\n\n    return encode_cyclic(encode_cyclic(s))\n'}], 'entry_point': 'decode_cyclic', 'test': "\n\nMETADATA = {}\n\n\ndef check(candidate):\n    from random import randint, choice\n    import string\n\n    letters = string.ascii_lowercase\n    for _ in range(100):\n        str = ''.join(choice(letters) for i in range(randint(10, 20)))\n        encoded_str = encode_cyclic(str)\n        assert candidate(encoded_str) == str\n\n"}

"""
