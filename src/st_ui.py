import streamlit as st
from PIL import Image
from main import get_answers_from_questions_and_documents, download_pretrained_model


def main():
	st.subheader("Document Understanding App 📑📊 ")
	with st.spinner(text="Downloading pretrained model..."):
		processor, model, device = download_pretrained_model()
	# Upload image through Streamlit file uploader
	uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"], key="fileuploader")

	# Display a placeholder for the image
	image_placeholder = st.empty()

	if uploaded_file is not None:
		# Display the uploaded image
		image = Image.open(uploaded_file)
		image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)

		# Display a placeholder for the document
		document_placeholder = st.empty()

		# Number of text boxes
		num_text_boxes = st.number_input('Insert a number', step=int(), value=1, max_value=6, min_value=1)

		# List to store user input
		user_inputs = []
		# Create and display text boxes
		for i in range(num_text_boxes):
			user_input = st.text_input(f"Question {i + 1}", key=f"Question_{i}")
			user_inputs.append(user_input)

		if st.button('get answers'):
			with st.spinner(text="In progress..."):
				answers = get_answers_from_questions_and_documents(device, processor, model, image, user_inputs)

			for answer in answers:
				f""""{answer['question']}  {answer['answer']}"""



if __name__ == "__main__":
	main()
