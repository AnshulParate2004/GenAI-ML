import json
from GenAI.client_config import client
from GenAI.prompt import system_prompt

def analyze_answer(history, question, answer, clarifications):
	prompt = f"""
	You are an interview evaluator.
	Here is the interview so far: {json.dumps(history, indent=2)}
	The current interview question is: "{question}"
	The candidate answered: "{answer}"
	Clarifications so far: {clarifications}

	1. Decide if the answer is clear.
	2. If unclear, suggest a follow-up sub-question.
	3. Score the quality of the answer from 1 to 10.

	Return JSON only:
	{{
	  "clear": true/false,
	  "sub_question": "string or null",
	  "score": number
	}}
	"""
	completion = client.chat.completions.create(
		model="llama-3.3-70b-versatile",
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": prompt}
		],
		response_format={"type": "json_object"},
		max_tokens=300,
		temperature=0.3,
	)
	raw = completion.choices[0].message.content
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		return {"clear": True, "sub_question": None, "score": 5}