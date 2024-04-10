import cv2
import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image


def download_pretrained_model():
	processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
	# important: we need to pad from the left when doing batched inference
	processor.tokenizer.padding_side = 'left'
	model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

	# move model to GPU if it's available
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model.to(device)
	return processor, model, device


def get_answers_from_questions_and_documents(device,processor, model, img: Image, questions: list) -> list:


	# prepare encoder inputs
	pixel_values = processor(img, return_tensors="pt").pixel_values
	batch_size = pixel_values.shape[0]

	# prepare decoder inputs
	task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
	prompts = [task_prompt.replace("{user_input}", question) for question in questions]
	decoder_input_ids = processor.tokenizer(prompts, add_special_tokens=False, padding=True,
	                                        return_tensors="pt").input_ids
	outputs = model.generate(
		pixel_values.to(device),
		decoder_input_ids=decoder_input_ids.to(device),
		max_length=model.decoder.config.max_position_embeddings,
		early_stopping=True,
		pad_token_id=processor.tokenizer.pad_token_id,
		eos_token_id=processor.tokenizer.eos_token_id,
		use_cache=True,
		num_beams=1,
		bad_words_ids=[[processor.tokenizer.unk_token_id]],
		return_dict_in_generate=True,
	)

	sequences = processor.batch_decode(outputs.sequences)
	answers = list()
	for seq in sequences:
		sequence = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
		sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
		answers.append(processor.token2json(sequence))
	cv2.destroyAllWindows()
	return answers


