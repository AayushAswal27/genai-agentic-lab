from langchain_text_splitters import CharacterTextSplitter

text = """Lionel Messi is one of the greatest football players in the history of the sport. He was born on June 24, 1987, in Rosario, Argentina, and started playing football at a very young age. At the age of 13, he moved to Spain and joined FC Barcelona's famous youth academy, La Masia.

Messi quickly became known for his incredible dribbling, speed, passing, vision, and goal-scoring ability. During his time at Barcelona, he won numerous trophies and became the club's all-time leading goalscorer. He has also won the Ballon d'Or multiple times, making him one of the most decorated players in football history.

With Argentina, Messi finally achieved his biggest international dream by winning the Copa América in 2021 and the FIFA World Cup in 2022. His performances in the 2022 World Cup were especially memorable, as he scored important goals and led Argentina to victory.

Messi's journey is an inspiration because he faced difficulties early in his life but never gave up on his dream. His talent, hard work, consistency, and humble personality have made him a legend loved by millions of football fans around the world.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_text(text)

print(result)