model_gpt = "gpt-3.5-turbo"
key = "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"

model_llama = "../model/llama/llama-2-7b-chat/ggml-model-f16_q4_0.bin"

question_types = [
    "clarification question", 
    "partial story state question", 
    "background knowledge question", 
    "next step hint",
    "counterfactual question",
    "probing question"
]

class_description = {
    "clarification question" : "question that makes sure the student has understood the problem, asking to rephrase part of the question or a detail in the problem",
    "partial story state question" : "question on a part of the problem, breaking down the problem into sub questions", 
    " background knowledge question" : "question about the background knowledge of the student, focused on the maths or the problem understanding", 
    "next step hint": "question about the next step to follow to solve the problem, checking the reasoning of the student",
    "counterfactual question" : "question that rewrites the problem to check the understanding of the student by seeing the change of the result",
    "probing question" : "question about the student's belief, checking for any misunderstanding of approximation"
}

challenge_description = {
    "conceptual understanding" : [
        "Facing problems involving abstract or unfamiliar concepts, leading to incorrect application of mathematical principles. Wrong understanding or application of the maths concepts",
        ["partial story state question"]
    ],
    "background knowledge" : [
        "Lack of necessary knowledge or information needed to solve the problem. Insufficient vocabulary, struggling with understanding mathematical terms and language used in the problem",
        ["background knowledge question"]
    ],
    "assumptions and presumptions": [
        "Making incorrect assumptions about the problem, leading to flawed reasoning. Unjustified extrapolation - extending known concepts or patterns to situations where they do not apply",
        ["probing question"]
    ],
    "calculation errors": [
        "Applying incorrect logic or methods to solve the problem, resulting in an erroneous solution. Decimal and Fraction Errors - Difficulty in performing accurate calculations involving decimals and fractions",
        ["next step hint"]
    ],
    "reasoning errors": [
        "Making mistakes in numerical calculations, leading to incorrect answers. Circular Reasoning - Getting trapped in a loop of flawed logic without reaching a solution",
        ["probing question", "next step hint", "partial story state question"]
    ],
    "generalization and application": [
        "Struggling to connect the problem to broader mathematical concepts or real-world applications, difficulty with abstraction. Contextual Blindness - Failing to recognize the significance of the problem in a broader mathematical or practical context",
        ["probing question", "counterfactual question"]
    ],
    "inefficiency in problem solving": [
        "Using a lengthy or convoluted method to solve a problem, wasting time and effort. Redundant Steps - Introducing unnecessary steps or calculations, complicating the problem-solving process further",
        ["next step hint", "partial story state question"]
    ],
    "inefficiency in problem solving": [
        "Selecting an inappropriate problem-solving approach, leading to inefficiency or incorrect solutions. Lack of Strategy Adaptation - Inability to switch strategies when the initially chosen one proves ineffective",
        ["partial story state question"]
    ]
}