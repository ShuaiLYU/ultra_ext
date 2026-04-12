"""
SpaCy subject extraction demo: root_noun vs first_chunk comparison.
200 test cases covering diverse bbox annotation patterns.

Usage:
    pip install spacy
    python -m spacy download en_core_web_sm
    python spacy_subject_demo.py
"""

import spacy

nlp = spacy.load("en_core_web_sm")


def extract_root_noun(doc) -> str:
    """Syntactic root noun: find ROOT, then its subject."""
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            break
    if root_token is None:
        return doc.text

    if root_token.pos_ in ("NOUN", "PROPN"):
        return _get_compound(root_token)

    if root_token.pos_ in ("VERB", "AUX"):
        for child in root_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return _get_compound(child)
        chunks = list(doc.noun_chunks)
        if chunks:
            return _get_compound(chunks[0].root)

    chunks = list(doc.noun_chunks)
    if chunks:
        return _get_compound(chunks[0].root)
    return doc.text


def extract_first_chunk(doc) -> str:
    """Simply return the first noun chunk's head noun."""
    chunks = list(doc.noun_chunks)
    if chunks:
        return _get_compound(chunks[0].root)
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            return _get_compound(token)
    return doc.text


def _get_compound(token) -> str:
    parts = []
    for child in token.children:
        if child.dep_ in ("compound", "amod") and child.i < token.i:
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)


if __name__ == "__main__":
    # (input_text, expected_head_noun_for_bbox)
    test_cases = [
        # ═══════════════════════════════════════════════════════
        # 1. PERSON + ACTION + OBJECT (bbox = person)
        # ═══════════════════════════════════════════════════════
        ("man walking a dog",                        "man"),
        ("woman holding a baby",                     "woman"),
        ("boy riding a bicycle",                     "boy"),
        ("girl eating an apple",                     "girl"),
        ("chef cooking in the kitchen",              "chef"),
        ("firefighter spraying water",               "firefighter"),
        ("old man sitting on a bench",               "man"),
        ("young woman carrying a bag",               "woman"),
        ("child playing with a ball",                "child"),
        ("man in a red shirt",                       "man"),
        ("player kicking a ball",                    "player"),
        ("driver in a truck",                        "driver"),
        ("worker fixing the roof",                   "worker"),
        ("person standing near the door",            "person"),
        ("someone holding an umbrella",              "someone"),
        ("dancer performing on stage",               "dancer"),
        ("soldier carrying a rifle",                 "soldier"),
        ("farmer feeding chickens",                  "farmer"),
        ("scientist looking through a microscope",   "scientist"),
        ("kid throwing a frisbee",                   "kid"),
        ("waiter serving food",                      "waiter"),
        ("nurse checking a monitor",                 "nurse"),
        ("teacher writing on a blackboard",          "teacher"),
        ("painter painting a wall",                  "painter"),
        ("singer holding a microphone",              "singer"),
        ("man reading a newspaper",                  "man"),
        ("woman talking on a phone",                 "woman"),
        ("boy climbing a tree",                      "boy"),
        ("girl drawing a picture",                   "girl"),
        ("man lifting weights",                      "man"),
        ("woman pushing a stroller",                 "woman"),
        ("man drinking coffee",                      "man"),
        ("woman typing on a laptop",                 "woman"),
        ("man grilling meat",                        "man"),
        ("woman watering plants",                    "woman"),
        ("man washing a car",                        "man"),
        ("child hugging a teddy bear",               "child"),
        ("athlete running on a track",               "athlete"),
        ("photographer taking a photo",              "photographer"),
        ("musician playing guitar",                  "musician"),

        # ═══════════════════════════════════════════════════════
        # 2. PERSON + LOCATION / CONTEXT (bbox = person)
        # ═══════════════════════════════════════════════════════
        ("man with glasses",                         "man"),
        ("woman in a blue dress",                    "woman"),
        ("the man on the left",                      "man"),
        ("man at the counter",                       "man"),
        ("woman behind the desk",                    "woman"),
        ("person in the background",                 "person"),
        ("man next to a car",                        "man"),
        ("woman near the window",                    "woman"),
        ("boy on the stairs",                        "boy"),
        ("girl by the pool",                         "girl"),
        ("man under an umbrella",                    "man"),
        ("woman on a balcony",                       "woman"),
        ("child in a playground",                    "child"),
        ("man in front of a building",               "man"),
        ("woman with a hat",                         "woman"),
        ("person at the bus stop",                   "person"),
        ("man wearing a suit",                       "man"),
        ("woman wearing sunglasses",                 "woman"),
        ("man with a backpack",                      "man"),
        ("baby in a car seat",                       "baby"),

        # ═══════════════════════════════════════════════════════
        # 3. GROUPS (bbox = group noun)
        # ═══════════════════════════════════════════════════════
        ("three people walking in the park",         "people"),
        ("two dogs playing on the beach",            "dogs"),
        ("couple sitting on a couch",                "couple"),
        ("group of students",                        "group"),
        ("family having dinner",                     "family"),
        ("crowd watching a game",                    "crowd"),
        ("team celebrating a goal",                  "team"),
        ("pair of shoes on the floor",               "shoes"),
        ("bunch of bananas",                         "bunch"),
        ("flock of birds in the sky",                "flock"),
        ("herd of cattle",                           "herd"),
        ("row of houses",                            "row"),
        ("group of people standing",                 "group"),
        ("two men talking",                          "men"),
        ("three cats on a roof",                     "cats"),

        # ═══════════════════════════════════════════════════════
        # 4. SHORT NOUN PHRASES (bbox = noun)
        # ═══════════════════════════════════════════════════════
        ("tall man",                                 "man"),
        ("red car",                                  "car"),
        ("big brown dog",                            "dog"),
        ("small wooden table",                       "table"),
        ("golden retriever puppy",                   "puppy"),
        ("white cat",                                "cat"),
        ("black horse",                              "horse"),
        ("blue bicycle",                             "bicycle"),
        ("green bottle",                             "bottle"),
        ("yellow flower",                            "flower"),
        ("old building",                             "building"),
        ("broken window",                            "window"),
        ("parked car",                               "car"),
        ("flying bird",                              "bird"),
        ("sleeping cat",                             "cat"),
        ("wooden fence",                             "fence"),
        ("metal gate",                               "gate"),
        ("glass door",                               "door"),
        ("stone wall",                               "wall"),
        ("leather bag",                              "bag"),

        # ═══════════════════════════════════════════════════════
        # 5. SINGLE NOUNS (bbox = exact word)
        # ═══════════════════════════════════════════════════════
        ("fireman",                                  "fireman"),
        ("couch",                                    "couch"),
        ("bicycle",                                  "bicycle"),
        ("dog",                                      "dog"),
        ("tree",                                     "tree"),
        ("car",                                      "car"),
        ("chair",                                    "chair"),
        ("laptop",                                   "laptop"),
        ("umbrella",                                 "umbrella"),
        ("backpack",                                 "backpack"),

        # ═══════════════════════════════════════════════════════
        # 6. ANIMALS + ACTION (bbox = animal)
        # ═══════════════════════════════════════════════════════
        ("dog chasing a cat",                        "dog"),
        ("cat sleeping on the sofa",                 "cat"),
        ("bird sitting on a branch",                 "bird"),
        ("horse running in the field",               "horse"),
        ("fish swimming in the tank",                "fish"),
        ("duck floating on water",                   "duck"),
        ("squirrel climbing a tree",                 "squirrel"),
        ("rabbit eating grass",                      "rabbit"),
        ("cow grazing in a meadow",                  "cow"),
        ("monkey hanging from a branch",             "monkey"),
        ("eagle flying over mountains",              "eagle"),
        ("bear catching a fish",                     "bear"),
        ("deer standing in the forest",              "deer"),
        ("dog lying on the floor",                   "dog"),
        ("cat looking out a window",                 "cat"),

        # ═══════════════════════════════════════════════════════
        # 7. OBJECTS + LOCATION (bbox = object)
        # ═══════════════════════════════════════════════════════
        ("a big red couch in the living room",       "couch"),
        ("the car on the left side",                 "car"),
        ("food on the table",                        "food"),
        ("flowers in a vase",                        "flowers"),
        ("painting hanging on the wall",             "painting"),
        ("laptop on the desk",                       "laptop"),
        ("bottle on the shelf",                      "bottle"),
        ("book on the nightstand",                   "book"),
        ("shoes by the door",                        "shoes"),
        ("hat on the rack",                          "hat"),
        ("clock on the wall",                        "clock"),
        ("lamp next to the bed",                     "lamp"),
        ("TV in the corner",                         "TV"),
        ("phone on the table",                       "phone"),
        ("bag under the chair",                      "bag"),
        ("plate on the counter",                     "plate"),
        ("towel hanging from a hook",                "towel"),
        ("mirror on the wall",                       "mirror"),
        ("picture above the fireplace",              "picture"),
        ("rug on the floor",                         "rug"),

        # ═══════════════════════════════════════════════════════
        # 8. VEHICLES + CONTEXT (bbox = vehicle)
        # ═══════════════════════════════════════════════════════
        ("car parked on the street",                 "car"),
        ("bus at the bus stop",                      "bus"),
        ("truck driving on the highway",             "truck"),
        ("bicycle leaning against a wall",           "bicycle"),
        ("motorcycle parked near the curb",          "motorcycle"),
        ("boat floating on the lake",                "boat"),
        ("airplane in the sky",                      "airplane"),
        ("train at the station",                     "train"),
        ("taxi in front of the hotel",               "taxi"),
        ("ambulance on the road",                    "ambulance"),

        # ═══════════════════════════════════════════════════════
        # 9. TRICKY / AMBIGUOUS (bbox = first entity)
        # ═══════════════════════════════════════════════════════
        ("man and woman dancing",                    "man"),
        ("baby in a stroller",                       "baby"),
        ("kids on a playground",                     "kids"),
        ("surfer on a wave",                         "surfer"),
        ("a person",                                 "person"),
        ("the big dog on the right",                 "dog"),
        ("a broken chair in the corner",             "chair"),
        ("an empty bottle on the ground",            "bottle"),
        ("a man and his dog",                        "man"),
        ("a woman with her child",                   "woman"),
        ("cat and dog playing together",             "cat"),
        ("mother holding her baby",                  "mother"),
        ("father teaching his son",                  "father"),
        ("bride and groom at the altar",             "bride"),
        ("pilot in the cockpit",                     "pilot"),

        # ═══════════════════════════════════════════════════════
        # 10. PASSIVE / UNUSUAL SYNTAX (bbox = main entity)
        # ═══════════════════════════════════════════════════════
        ("a cake decorated with flowers",            "cake"),
        ("a wall covered in graffiti",               "wall"),
        ("a road lined with trees",                  "road"),
        ("a child surrounded by toys",               "child"),
        ("a dog wrapped in a blanket",               "dog"),
        ("pizza topped with mushrooms",              "pizza"),
        ("a field full of sunflowers",               "field"),
        ("a shelf filled with books",                "shelf"),
        ("a man dressed in black",                   "man"),
        ("a car covered in snow",                    "car"),

        # ═══════════════════════════════════════════════════════
        # 11. WITH DETERMINERS/QUANTIFIERS (bbox = stripped noun)
        # ═══════════════════════════════════════════════════════
        ("the man",                                  "man"),
        ("a woman",                                  "woman"),
        ("an elephant",                              "elephant"),
        ("some flowers",                             "flowers"),
        ("this building",                            "building"),
        ("that car over there",                      "car"),
        ("several people crossing the street",       "people"),
        ("many birds on a wire",                     "birds"),
        ("one child standing alone",                 "child"),
        ("another dog behind the fence",             "dog"),

        # ═══════════════════════════════════════════════════════
        # 12. COMPOUND NOUNS (bbox = compound)
        # ═══════════════════════════════════════════════════════
        ("fire truck on the road",                   "truck"),
        ("police car with lights on",                "car"),
        ("traffic light at the intersection",        "light"),
        ("street light at night",                    "light"),
        ("dining table set for dinner",              "table"),
        ("coffee cup on the desk",                   "cup"),
        ("baseball bat leaning on a wall",           "bat"),
        ("tennis racket on the ground",              "racket"),
        ("cell phone on the table",                  "phone"),
        ("swimming pool in the backyard",            "pool"),
    ]

    root_correct = 0
    first_correct = 0
    total = len(test_cases)

    print(f"{'INPUT':<50} {'EXPECTED':<15} {'ROOT_NOUN':<25} {'FIRST_CHUNK':<25} {'R':>2} {'F':>2}")
    print("=" * 130)

    disagreements = []

    for text, expected in test_cases:
        doc = nlp(text)
        root_result = extract_root_noun(doc)
        first_result = extract_first_chunk(doc)

        root_ok = expected.lower() in root_result.lower()
        first_ok = expected.lower() in first_result.lower()

        root_correct += root_ok
        first_correct += first_ok

        rm = "✓" if root_ok else "✗"
        fm = "✓" if first_ok else "✗"

        print(f"{text:<50} {expected:<15} {root_result:<25} {first_result:<25} {rm:>2} {fm:>2}")

        if root_result != first_result:
            disagreements.append((text, expected, root_result, first_result, root_ok, first_ok))

    print("=" * 130)
    print(f"\nSCORES:")
    print(f"  root_noun:   {root_correct}/{total} = {100*root_correct/total:.1f}%")
    print(f"  first_chunk: {first_correct}/{total} = {100*first_correct/total:.1f}%")
    print(f"  total cases: {total}")

    if disagreements:
        print(f"\nDISAGREEMENTS ({len(disagreements)} cases):")
        print(f"  {'INPUT':<50} {'EXPECTED':<12} {'ROOT_NOUN':<20} {'R':>2}  {'FIRST_CHUNK':<20} {'F':>2}")
        print(f"  {'-'*120}")
        for text, expected, root_r, first_r, rok, fok in disagreements:
            rm = "✓" if rok else "✗"
            fm = "✓" if fok else "✗"
            print(f"  {text:<50} {expected:<12} {root_r:<20} {rm:>2}  {first_r:<20} {fm:>2}")

        root_wins = sum(1 for *_, rok, fok in disagreements if rok and not fok)
        first_wins = sum(1 for *_, rok, fok in disagreements if fok and not rok)
        both_wrong = sum(1 for *_, rok, fok in disagreements if not rok and not fok)
        print(f"\n  root_noun wins: {root_wins}  |  first_chunk wins: {first_wins}  |  both wrong: {both_wrong}")