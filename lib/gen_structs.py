
class GenerateOutput():
    def __init__(
        self, input_text: str, output_text: str, reason_text: str, gen_time: float, 
        input_token_cnt: int, output_token_cnt: int, thoughts_token_cnt: int
    ):
        self.input_text = input_text
        self.output_text = output_text
        self.reason_text = reason_text
        self.gen_time = gen_time
        self.input_token_cnt = input_token_cnt
        self.output_token_cnt = output_token_cnt
        self.thoughts_token_cnt = thoughts_token_cnt

    def printout(self):
        print("=== Generation Output ===")
        print(f"Input Text: {self.input_text}")
        print(f"Reasoning Text: {self.reason_text}")
        print(f"Output Text: {self.output_text}")
        print(f"Generation Time: {self.gen_time:.2f} seconds")
        print(f"Input Token Count: {self.input_token_cnt}")
        print(f"Output Token Count: {self.output_token_cnt}")
        print(f"Thoughts Token Count: {self.thoughts_token_cnt}")
