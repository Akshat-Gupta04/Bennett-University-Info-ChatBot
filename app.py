import chainlit as cl
from src.llm import create_faiss_database, load_faiss_database, create_retrieval_qa_chain

db_path="bennettUniversityChatBot_with_chainlit/vectordataset"
URLs=[
    "https://www.bennett.edu.in/","https://www.bennett.edu.in/about-us/overview/","https://www.bennett.edu.in/programs/","https://www.bennett.edu.in/placements/","https://www.bennett.edu.in/bennett-life/","https://www.bennett.edu.in/innovation-centre/",
    "https://alumni.bennett.edu.in/","https://www.bennett.edu.in/career","https://www.bennett.edu.in/admission/scholarships/","https://www.bennett.edu.in/admission/scholarships/#btech_cse"
]



try:
  
    faiss_db = load_faiss_database(db_path)
except:
    faiss_db = create_faiss_database(URLs, db_path)


qa_chain = create_retrieval_qa_chain(faiss_db)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Campus life",
            message="Can you help me by telling what campus life feels in bennett feels?",
            ),

        cl.Starter(
            label=" Placements in Bennett",
            message="tell about placements in bennett uiniversity in tabular format? ",
            ),
        cl.Starter(
            label="Courses Offered",
            message="tell which courses i can avail in bennett university?",
            ),
        cl.Starter(
            label="About Bennett",
            message="tell me about bennett university ?",

            )
        ]




@cl.on_message
async def main(user_message: cl.Message):
    user_text = user_message.content.strip()
    result = qa_chain.run(user_text)
    await cl.Message(content=result).send()
