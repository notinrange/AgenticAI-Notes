from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Optional,Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# schema
class Review(BaseModel):
    keythemes: list[str] = Field(description="Write down all the key themes mentioned in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review")
    pros: Optional[list[str]] = Field(default=None, description="List the pros mentioned in the review")
    cons: Optional[list[str]] = Field(default=None, description="List the cons mentioned in the review")
    name: Optional[str] = Field(default=None, description="Name of the reviewer")


structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""I purchased this portable bed table six months ago and it has completely transformed my daily routine. Whether I am working from home, having breakfast in bed, or reading late at night, this table has proven to be an absolute game changer.

The adjustable height feature is brilliant, allowing me to set it perfectly whether I am sitting up or lying down. The surface is sturdy enough to hold my laptop, a cup of tea, and my notebook simultaneously without any wobbling at all.

Assembly took less than ten minutes with no tools required. The foldable legs make storage incredibly convenient, sliding right under my bed when not in use. After six months of daily use, there is no sign of wear or weakness in the joints.

The build quality feels solid and premium despite the affordable price. The non-slip surface is a thoughtful addition that keeps everything in place. I also love how lightweight it is, making it easy to move from room to room.

My only minor complaint is that the surface could be slightly wider for larger laptops. A small lip or edge to prevent things from sliding off would also be a welcome addition.

Overall this is one of the best small purchases I have made this year. If you spend any significant time working or relaxing in bed, this portable table is absolutely worth every rupee.""")


print(result.name)