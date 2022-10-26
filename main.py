"""
This is a hello world add-on for DocumentCloud.

It demonstrates how to write a add-on which can be activated from the
DocumentCloud add-on system and run using Github Actions.  It receives data
from DocumentCloud via the request dispatch and writes data back to
DocumentCloud using the standard API
"""

from documentcloud.addon import AddOn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Flan(AddOn):
    """An example Add-On for DocumentCloud."""

    def main(self):
        """The main add-on functionality goes here."""
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


        # fetch your add-on specific data

        self.set_message("Trying to run the Flan model on your documents!")

        # add a hello note to the first page of each selected document
        for document in self.get_documents():
            # Get the text from the document
            prompt = self.data.get("prompt")
            input_text = prompt + document.text
            print(input_text)
            # get_documents will iterate through all documents efficiently,
            # either selected or by query, dependeing on which is passed in
            # the request.
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            print(tokenizer.decode(outputs[0]))
            document.description(tokenizer.decode(outputs[0]))
            with open("summary.txt", "w+") as file_:
                file_.write("Hello world!")
                self.upload_file(file_)

    

if __name__ == "__main__":
    Flan().main()
