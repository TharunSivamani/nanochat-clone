"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("cais/mmlu", name=subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train":
            """
            {
                "answer": 1,
                "choices": [
                    "Adams only.",
                    "Brooks only.",
                    "Case only.",
                    "Adams and Brooks"
                ],
                "question": "Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were",
                "subject": ""
            }
            """
            self.ds = self.ds.map(lambda row: row['train'], remove_columns=["train"])
        
    @property
    def eval_type(self):
        return 'categorical'
    
    def num_examples(self):
        return len(self.ds)
    
    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        subject = row["subject"]
        assert len(choices) == 4, "MMLU should have 4 choices"
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]

        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }

        return conversation
    
    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message

"""

python -m tasks.common
Length of MMLU: 99842

5th example is: {'messages': [{'role': 'user', 'content': 'Multiple Choice question: The color of what is an inherited characteristic?\n- vision organ=A\n- wallpaper=B\n- blood=C\n- dirt=D\n\nRespond only with the letter of the current answer.'}, {'role': 'assistant', 'content': 'A'}], 'subject': '', 'letters': ('A', 'B', 'C', 'D')}

Length of sliced MMLU[5:10]:  5

0th example of sliced MMLU:  {'messages': [{'role': 'user', 'content': 'Multiple Choice question: The color of what is an inherited characteristic?\n- vision organ=A\n- wallpaper=B\n- blood=C\n- dirt=D\n\nRespond only with the letter of the current answer.'}, {'role': 'assistant', 'content': 'A'}], 'subject': '', 'letters': ('A', 'B', 'C', 'D')}

They match:  True

10th Example of the sliced MMLU for viz:  {'messages': [{'role': 'user', 'content': "Multiple Choice question: India has a very vast history and heritage. The country has witnessed some of the most important events in the past and every event has an effect of its own on the culture, religion, lifestyle, and economy of the country. This is one of the leading reasons why tourists prefer India to travel and explore. Historical tours in India provide everything that a traveler expects from his/her journey. These historical tours give an opportunity to study and explore Indian history. Historical tours of India generally consisted of four phases: ancient, medieval , modern India and contemporary India. Some of the historical travel packages are: Forts and Palaces The rich cultural past of India is reflected in the countless forts and palaces across the country. Most of the forts were constructed as a defense device to keep the enemy away and palaces were made on the names of the kings and queens. Some of the famous forts and palaces are: Agra Fort (Agra), Fatehpur Sikri (Agra), Hawa Mahal (Jaipur), Amber Fort (Jaipur), City Palace (Jaipur), Gwalior Fort (Gwalior), Mysore Palace (Mysore), Red Fort (Delhi), etc. India Heritage Tours The main purpose of India Heritage tours is to make Indians as well as tourists familiar with the rich Indian legacy . The heritage of the country is at least 5000 years. Some of the heritage tours are: medieval heritage, Rajput heritage, Golden Triangle, splendid Gujarat, etc. South India Historical Tours The Southern India is known for its diversity and there are many places, which attract tourists, not only from the country but also from all around the world. The famous tourists' destinations in South India include Goa, Cochin, Kerala, Munnar, Periyar, Kottayam and the list goes on and on. Most forts were built in order to\n- honor the kings=A\n- act as a palace for queens=B\n- keep enemies away=C\n- attract tourists=D\n\nRespond only with the letter of the current answer."}, {'role': 'assistant', 'content': 'C'}], 'subject': '', 'letters': ('A', 'B', 'C', 'D')}

"""