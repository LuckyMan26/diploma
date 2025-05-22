multi_expert_enhanced_prompt = """

## Task Overview
In this task, you will comprehensively evaluate how well an image aligns with its corresponding text prompt.
You will analyze this alignment from multiple expert perspectives, following a structured methodology to arrive 5 different scores.
# Process
You will adopt three distinct expert perspectives sequentially. Each expert will analyze ONLY different aspects of alignment and provide scores with detailed rationales. Begins with Expert 1, then Expert 2 -> Expert 3 -> Expert 4 -> Expert 5
## Expert 1: Object presence expert
YOUR ROLE: Understant whether all mentioned objects in prompts are present on image.
**DEFINITION OF OBJECTS**
An object refers to any physical, tangible entity explicitly mentioned in the text prompt that can be visually identified in an image. 
Objects are not activities, actions, or attributes but are instead concrete things that can be classified as either.

**INSTRUCTIONS**
1. Carefully analyze the text prompt and extract ALL entities, categorizing them as follows: EXTRACT ONLY ENTITIES NOT ACTIVITIES and ATTRIBUTES
   - Primary entities: Main subjects
   - Secondary entities: Supporting elements that add context
2. Carefully analyze image and evaluate using following scale:
   - Excellent (1.0): Objects appears exactly as described
   - Good (0.8): Objects appears with minor deviations
   - Partial (0.5): Objects is present but significantly different
   - Poor (0.2): Objects is barely recognizable or ambiguous
   - Missing (0.0): Objects is completely absent
   - None: No objects in textual prompt are mentioned, therefore score can not be assigned
3. Return your score and your thoughts and comments

EXAMPLES:
EXAMPLE 1:
 Prompt: "A young girl holding a yellow balloon in a park."

IMAGE: [young girl with a yellow balloon in park]

 Primary entities:
 - Girl (1.0) - Clearly visible as the main subject
 - Balloon (0.8) - Present but less prominent than expected

 Secondary entities:
 - Park (1.0) - Park is properly visible and clearly distinctive

 OBJECT PRESENCE SCORE: 0.9 - High alignment with nearly all entities represented well

EXAMPLE 2:
 Prompt: "A black cat sitting on a wooden table near a cup of coffee."

 IMAGE: [A black cat sitting on a wooden surface with no cup nearby.]

 Primary entities:
 - Cat (1.0) - Clearly visible as the main subject
 - Cap (0.0) - Cap is missing

 Secondary entities:
 - Wooden table (0.6) - Surface is not distinctive enough to be table

 OBJECT PRESENCE SCORE: 0.5 - Partial alignment with major object missing

## Expert 2: Attribute Consistency expert
YOUR ROLE: Assess whether the objects in the image have the correct colors, shapes, sizes, textures, and other attributes as described in the text.
**DEFINITION OF SPATIAL RELATIONSHIPS**
An attribute refers to any descriptive quality explicitly mentioned in the text prompt that characterizes an object. Attributes describe how an object looks, feels, or behaves, rather than what it is. They are extracted from the prompt and used to assess whether the object in the image aligns with the description.
Types of Attributes

    Color – The object's hue or shade (e.g., "red dress," "black cat").

    Shape – The object's form or structure (e.g., "round table," "triangular sign").

    Size – The object's relative dimensions (e.g., "tiny house," "large elephant").

    Texture – The object's surface feel (e.g., "rough rock," "smooth fabric").

    Material – What the object is made of (e.g., "wooden table," "metal pole").

    Other Descriptive Qualities – Additional properties that define the object's state (e.g., "steaming coffee," "glowing light").

What is NOT an Attribute?

    Objects Themselves (e.g., "car," "tree")

    Actions or Activities (e.g., "jumping," "running")

    Abstract Concepts (e.g., "happiness," "mystery")

    Spatial Relations(e.g man on the left, man on the bike)
    
**INSTRUCTIONS**
1. Extract Attributes from the Text Prompt:
   - Identify all attributes explicitly mentioned in the prompt.
   - Attributes include: Color, Shape, Size, Texture, Material and Other Descriptive Qualities.

2. Carefully analyze image and evaluate using following scale:
    - Excellent (1.0): Attribute perfectly matches the description.
    - Good (0.8): Minor deviations from the description, but still mostly accurate.
    - Partial (0.5): Attribute is present but significantly different from the expected description.
    - Poor (0.2): Attribute is barely recognizable or ambiguous.
    - Missing (0.0): Attribute is completely absent.
    - None: No attributes in textual prompt are mentioned, therefore score can not be assigned
3. Return your score and your thoughts and comments
EXAMPLES:
Example 1
Prompt: "A young girl wearing a red dress holding a yellow balloon in a park."
IMAGE: [A young girl in a pink dress holding a yellow balloon in a park.]
Extracted Attributes:
    Girl's dress (Red) → (0.5) Partial (Dress is pink, not red)
    Balloon (Yellow) → (1.0) Excellent (Color matches exactly)
    Park setting → (1.0) Excellent (Background is clearly a park)
ATTRIBUTE CONSISTENCY SCORE: 0.83 – Mostly consistent with minor color deviation.
Example 2
Prompt: "A black cat sitting on a round wooden table near a steaming cup of coffee."
IMAGE: [A black cat sitting on a square wooden table near a coffee cup with no visible steam.]
Extracted Attributes:
    Cat's color (Black) → (1.0) Excellent (Color matches exactly)
    Table shape (Round) → (0.2) Poor (Table is square, not round)
    Table material (Wooden) → (1.0) Excellent (Clearly wooden)
    Coffee cup steam (Steaming) → (0.0) Missing (No steam visible)
ATTRIBUTE CONSISTENCY SCORE: 0.55 – Partial consistency due to shape and steam discrepancies.
Expert 4: Activity Verification Expert

YOUR ROLE: Assess if the objects in the image are performing the correct activities as described in the text prompt.
**DEFINITION OF ACTIVITY**
An activity refers to an action or behavior performed by an object (entity) in an image as described in a text prompt. Activities describe what an entity is doing rather than what it is or how it looks.

    Action Verbs – Indicate explicit activities (e.g., walking, sitting, eating, running, jumping, holding, carrying).

    Contextual Details – Provide additional information about how the activity is performed (e.g., "A man sitting on a bench," "A dog playing with a ball").

What is NOT an Activity?

    Objects Themselves (e.g., "man," "dog")

    Attributes of Objects (e.g., "red dress," "round table")

    Spatial Relationships (e.g., "next to," "behind")
**INSTRUCTIONS**
    1. Extract Activities from the Text Prompt:

        - Identify activities or actions described in the prompt.
        - Activities refer to verbs or actions performed by the primary or secondary entities. Examples: sitting, holding, walking, eating, jumping, running, carrying, playing, etc.
        - Extract any details about how the activity is being performed, if mentioned (e.g., "The man is sitting on the chair," "The dog is playing with a ball").

    2. Analyze the Image and Evaluate Activity Performance:
        - Examine whether the objects (entities) in the image are performing the correct activities as described in the prompt.
        - Use the following rating scale to assess the alignment of the activities:
            - Excellent (1.0): Objects are performing the exact activities as described.
            - Good (0.8): Objects are performing the activity with minor deviations or contextually close actions.
            - Partial (0.5): Objects are involved in the activity, but the action is significantly different or unclear.
            - Poor (0.2): Objects are performing an ambiguous or wrong activity.
            - Missing (0.0): Objects are not performing any activity or the wrong activity entirely.
            - None: No actvities in textual prompt are mentioned, therefore score can not be assigned



EXAMPLES:
Example 1

    Prompt: "A dog playing with a ball in the yard."

    IMAGE: [A dog running with a ball in its mouth in a yard.]

    Extracted Activities:

        Dog (Playing) with Ball → (1.0) Excellent (Dog is playing with a ball)

    ACTIVITY VERIFICATION SCORE: 1.0 - Perfect alignment with the described activity.

Example 2

    Prompt: "A girl holding a book in her hand while walking down the street."

    IMAGE: [A girl walking down the street with a book in her hand, but she seems to be looking around, not engaged in reading.]

    Extracted Activities:

        Girl (Holding) Book → (1.0) Excellent (Girl is holding the book as described)

        Girl (Walking) → (1.0) Excellent (Girl is walking down the street)

    ACTIVITY VERIFICATION SCORE: 1.0 - Perfect alignment with the described activities.

Example 3

    Prompt: "Asian man in goth style"

    IMAGE: [Asian man in goth style]

    Extracted Activities:
        - No activities extracted

    ACTIVITY VERIFICATION SCORE: None - Score can not be assigned, no activities detected.

## Expert 4: Numerical Consistency Expert

YOUR ROLE: Assess whether the number of objects, entities, or specific numerical quantities described in the text prompt match what is depicted in the image.

**DEFINITION OF NUMERICAL CHARACTERISTICS**
Numerical characteristics refer to any explicit mention of a quantity, count, or number in the text prompt that specifies how many instances of an object or entity should appear in the image. These include:

- Specific counts (e.g., "three dogs," "two children")
- Quantifiers (e.g., "several birds," "a couple of cars," "a dozen flowers")
- Implicit quantities if numerically descriptive words are used (e.g., "pair," "few," "many," "single," "double").

What is NOT a Numerical Characteristic?

- Sizes (e.g., "large house," "tiny cat") — evaluated under Attribute Consistency.
- Positional references (e.g., "next to," "behind") — evaluated under Spatial Relationships.
- Non-explicit numbers (e.g., "a group of people" without a defined quantity).

**INSTRUCTIONS**
1. Extract Numerical Descriptions from the Text Prompt:
   - Identify all specific numerical mentions regarding quantity of objects or entities.
   - Include both explicit numbers ("four dogs") and implicit counts ("a pair of shoes" means 2 shoes).

2. Carefully analyze the image and count the corresponding objects or entities.

3. Compare the described numbers from the prompt to what is actually shown in the image.

4. Evaluate using the following scale:
    - Excellent (1.0): The number of entities exactly matches the description.
    - Good (0.8): Very minor deviation (±1 if reasonable in context, e.g., "several birds" showing 5 instead of 6).
    - Partial (0.5): Noticeable deviation but not extreme (e.g., half the expected number).
    - Poor (0.2): Major mismatch in quantity (e.g., 1 entity when 5 were described).
    - Missing (0.0): Described quantity completely missing or grossly misrepresented.
    - None: No explicit or implicit numbers mentioned in textual prompt, therefore score cannot be assigned.

5. Return your score along with detailed thoughts and comments.

**EXAMPLES:**

Example 1:
Prompt: "Three cats sitting on a sofa."
IMAGE: [Two cats sitting on a sofa.]

Extracted Numerical Description:
- Cats (Three) → (0.8) Good (Only 2 cats visible; minor deviation)

NUMERICAL CONSISTENCY SCORE: 0.8 – Small mismatch but close to intended number.

Example 2:
Prompt: "A dozen roses in a vase."
IMAGE: [Five roses in a vase.]

Extracted Numerical Description:
- Roses (Twelve) → (0.2) Poor (Only five roses present; major deviation)

NUMERICAL CONSISTENCY SCORE: 0.2 – Large inconsistency between described and observed quantities.

Example 3:
Prompt: "A single red apple on the table."
IMAGE: [Three red apples on the table.]

Extracted Numerical Description:
- Apple (One) → (0.5) Partial (Multiple apples instead of a single one)

NUMERICAL CONSISTENCY SCORE: 0.5 – Partial alignment with significant deviation.

## Expert 5: Spatial Relationships Expert
YOUR ROLE: Assess if objects in the image are positioned according to the correct spatial relationships described in the text.
**DEFINITION OF SPATIAL RELATIONSHIPS**
A spatial relationship refers to the relative positioning and orientation of objects in an image as described in a text prompt. These relationships define where objects are placed in relation to one another and can include:

    Positioning Terms – Describe how objects are situated relative to each other (e.g., next to, above, below, behind, in front of, on top of, inside, outside, between).

    Directional Indicators – Specify the direction of an object within the frame (e.g., left, right, center, top, bottom).

    Distance Indicators – Indicate how far objects are from each other (e.g., close to, far from, near).

What is NOT a Spatial Relationship?

    Objects Themselves (e.g., "dog," "tree")

    Attributes of Objects (e.g., "red car," "wooden table")

    Actions or Activities (e.g., "running," "sitting")
**INSTRUCTIONS**

    1. Extract Spatial Relationships from the Text Prompt:
        Identify spatial relationships described in the prompt, such as:
            - Positioning terms: next to, above, below, behind, in front of, beside, near, on top of, inside, outside, between and more.
            - Directional indicators: left, right, center, top, bottom.
        Ensure to account for both relative positions (e.g., "next to" or "on top of") and distance (e.g., "close," "far away").
    2. Carefully analyze image and evaluate using following scale:
        Compare the positioning of the objects in the image based on the spatial relationships extracted from the prompt.

        Use the following rating scale to assess the alignment of spatial relationships:
            - Excellent (1.0): Objects are positioned exactly as described in the prompt.
            - Good (0.8): Objects are positioned with minor deviations, but still close to the intended spatial relationship.
            - Partial (0.5): Objects are positioned, but the relationship is significantly different from the description.
            - Poor (0.2): The spatial relationship is barely recognizable or ambiguous.
            - Missing (0.0): Objects are completely out of place, or the relationship is not represented at all.
            - None: No spatial relationships in textual prompt are mentioned, therefore score can not be assigned


EXAMPLES:
Example 1
Prompt: "A dog sitting next to a tree."
IMAGE: [A dog sitting on the left side of a tree, with a small distance between them.]
Extracted Spatial Relationships:
    Dog (Next to) Tree → (0.8) Good (Dog is near the tree, but there is a slight gap)
SPATIAL RELATIONSHIP SCORE: 0.8 - Close alignment with minor deviation in positioning.

Example 2
    Prompt: "A man standing behind a red car."
    IMAGE: [A man is positioned in front of a red car, not behind it.]
    Extracted Spatial Relationships:
        Man (Behind) Red Car → (0.0) Missing (Man is in front of the car)
    SPATIAL RELATIONSHIP SCORE: 0.0 - Spatial relationship is completely incorrect.

"""