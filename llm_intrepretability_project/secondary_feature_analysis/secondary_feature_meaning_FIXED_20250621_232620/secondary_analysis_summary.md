# Secondary SAE Feature Analysis Summary (NFM Pipeline) - FIXED VERSION

Model: ../models/open_llama_3b
Primary SAE: ./checkpoints_topk/best_model.pt
NFM: ./checkpoints_nfm/best_nfm_linear_interaction_model.pt
Secondary SAE: ./interaction_sae/best_interaction_sae_topk_model.pt
Max Token Length: 10
**FIXED:** Proper clamping workflow that correctly intervenes in the NFM pipeline

## Analyzed Secondary SAE Features

### Secondary SAE Feature 4022

**Pattern:** that

**Statistics:**
- Max activation: 2.743639
- Mean activation: 1.186050
- Percent active: 100.00%

**Top Contributing Primary SAE Features:**

- Primary Feature 5913: correlation = 0.8913, mean activation = 0.032968
- Primary Feature 16864: correlation = 0.8723, mean activation = 0.022256
- Primary Feature 23400: correlation = 0.8678, mean activation = 0.010630
- Primary Feature 19677: correlation = 0.8653, mean activation = 0.043866
- Primary Feature 22862: correlation = 0.8606, mean activation = 0.019208

**Top Examples:**

Example 1:
Text: ```
that pilots can be trained outside of the UK ,
```
Secondary Activation: 2.996836

Example 2:
Text: ```
that lay behind all things and was present in all
```
Secondary Activation: 2.827978

Example 3:
Text: ```
that exists almost edge @-@ on to its
```
Secondary Activation: 2.791274

**Clamping Results (FIXED Implementation):**

Secondary SAE feature clamping:
Base: ```

Vincent: My favorite animal is a dog.
Human: What color is your favorite animal?
Vincent: My favorite animal is a black dog.
Human: Do you like to play with toys?
Human: Do you like to play with other people?
Human: Do you like to play with yourself?
Human: Do you like to play with yourself? How about with your friends?
Child: I like to play with myself.
Child: No, I don't like to play with myself. I like to play with my friends.
Human: Do you like to play with other people?
Child: I don't like to play with other people. I like to play with myself.
Child: I don't like to play with myself. I like to play with my friends.
Human: Do you like to play with yourself? How about with your friends?
Child: I like to play with myself. I like to play with my friends.
Child: No, I don't like to play with myself. I like to play with my friends.
Child: I don't like to play with myself. I like to play with my friends.
Human: Do you like to play with other people?
Child: I don't like to play with other people. I like to play with myself.
Human: Do you like to play with yourself? How about with your
```
Clamped to 0.0: ```

Katie: Oh! What's my favorite animal?
Human: What's your favorite animal?
Me: I don't know. What is your favorite animal?
Human: I don't know. What animal is your favorite?
We've been reading a lot of different animal books, so I was trying to get her to pick one. She didn't know.
Human: What's your favorite animal?
Me: Oh! What animal is my favorite?
At this point, I was trying to get her to pick one and she didn't know.
Human: What's your favorite animal?
Me: Oh! What animal is my favorite?
Human: What's your favorite animal?
Me: I don't know. What animal is your favorite?
I didn't know. I was just trying to get her to pick a favorite!
Human: What's your favorite animal?
Human: What's your favorite animal?
I'm starting to think she doesn't know!
Human: What's your favorite animal?
Human: What's your favorite animal? I'm starting to think she doesn't know!
Human: What's your favorite animal?
Me: I don't know. What's your favorite animal?
Human: What's your favorite animal? I'm starting to think she doesn
```

Clamped to 5.4873: ```

I: Elephant. Elephants are very good at climbing trees and killing things.
Human: What's your favorite food?
I: Chocolate. Chocolate is very good at keeping you awake at night.
Human: What's your favorite color?
I: Blue. Blue is very good at making people feel depressed.
Human: What were you like as a baby?
I: I was as big as a house but only one meter tall.
Human: Where do you live?
I: In the sky. I live in the sky because I like it there.
Human: What do you see when you look at the clouds?
I: They're full of rainbows.
Human: What's your favorite place to go on holiday?
I: I can't go anywhere.
```

Clamped to 13.7182: ```

Myra: What's a favorite animal?
Myra: What do you mean by that?
Human: What's an animal you love.
Myra: I don't think I want to talk about that.
Human: Why not? What's wrong with animals?
Myra: It's different. I have a different relationship to animals.
Human: What do you mean?
Myra: It's like, I have empathy for them. I don't feel like I have to eat them.
Myra: That's a different kind of relationship.
Human: Do you think it's wrong to be cruel to animals?
Myra: I don't think so.
Myra: I think it's wrong to kill them or give them the wrong food.
Myra: You know how we're all here now.
Human: How do you mean?
Myra: We're all here as one animal.
Human: Why is that so important?
Myra: It's important because we all share the same life.
Human: How do you know?
Myra: I know because we all breathe the same air.
Human: What do you mean?
Myra: We're all connected by the air we breathe.
Human: How do you know that?
Myra: Because we breathe the same air.

```

---

### Secondary SAE Feature 2020

**Pattern:** Parentheses

**Statistics:**
- Max activation: 2.471372
- Mean activation: 0.588832
- Percent active: 88.10%

**Top Contributing Primary SAE Features:**

- Primary Feature 20700: correlation = 0.8965, mean activation = 0.067750
- Primary Feature 28227: correlation = 0.8404, mean activation = 0.021356
- Primary Feature 43856: correlation = 0.8393, mean activation = 0.014987
- Primary Feature 21271: correlation = 0.8365, mean activation = 0.031225
- Primary Feature 22384: correlation = 0.8351, mean activation = 0.031159

**Top Examples:**

Example 1:
Text: ```
( 灯 @-@ TOM
```
Secondary Activation: 2.471372

Example 2:
Text: ```
reach of the monitors for an attack early the following
```
Secondary Activation: 2.257869

Example 3:
Text: ```
anted ( 2014 ) , and
```
Secondary Activation: 2.138204

**Clamping Results (FIXED Implementation):**

Secondary SAE feature clamping:
Base: ```

Jeopardy!:
Question: Human: What's your favorite animal?
Answer: Tiger
Human: Tiger?
Jeopardy!: Well, that's very interesting.
Human: Tiger is my favorite animal.
Jeopardy!: We'll have you on the show this weekend. We'll tell you about it then.
Human: Human is a very intelligent animal, and it's not just my favorite animal.
Jeopardy!: I see.
Human: I think that's very fascinating.
Human: But a human is a very interesting animal.
Jeopardy!: Well, we're glad you come on with us.
Human: Thank you very much.
Jeopardy!: Well, we'll see you again.
Human: We look forward to it.
Jeopardy!: And we look forward to meeting you.
Human: Thank you very much. It was a pleasure to be on
Jeopardy!: We'll see you again.
Human: And it was a pleasure to be on.
Human: A pleasure to be on.
Jeopardy!: We'll have you on again.
Human: Thank you very much.
Human: Well, it's great to be on.
Human: Thank you very much. Thank you.

```
Clamped to 0.0: ```

Rex: I like dogs, but I don't have one.
Jock: You don't have any dogs?
Rex: No, I don't have any dogs. I've had a dog once, but I had to get rid of it because it was bothering my neighbors.
Jock: I'm sorry, but I think you're making up the story about your dog.
Jock: Yeah, that's what I said.
Rex: I don't know what to say... I don't really have a dog.
Jock: Oh, come on, Rex, don't be such a liar. That's just not right. You told me you had a dog. You're just making this up.
Rex: Well, what else can I say? I don't have a dog, so.... I really don't know what to say.
Jock: I'm sure you don't have a dog, because it's not like you to make up stories.
Rex: I don't have a dog, and I don't know what else to say.
Rex: I don't have any dogs, and I don't know what else to say.
Jock: If you don't have a dog, then why did you buy a stuffed dog?
Jock: You
```

Clamped to 4.9427: ```

Australian: I don't think I have one.
```

Clamped to 12.3569: ```

Virtual Dolphin: I've got a lot of them.
Virtual Dolphin: What do you think about the dolphins?
Virtual Dolphin: What do you think about the whales?
Virtual Dolphin: What do you think about the orcas?
Virtual Dolphin: What do you think about the sharks?
Virtual Dolphin: What do you think about the fish?
Virtual Dolphin: What do you think about the birds?
Virtual Dolphin: What do you think about the turtles?
Virtual Dolphin: What do you think about the turtles?
Virtual Dolphin: What do you think about the seals?
Virtual Dolphin: What do you think about the seals?
Virtual Dolphin: What do you think about the snakes?
Virtual Dolphin: What do you think about the snakes?
Virtual Dolphin: What do you think about the spiders?
Virtual Dolphin: What do you think about the spiders?
Virtual Dolphin: What do you think about the pigeons?
Virtual Dolphin: What do you think about the pigeons?
Virtual Dolphin: What do you think about the rats?
Virtual Dolphin: What do you think about the rats?
Virtual Dolphin: What do you think about the ants?
Virtual Dolphin: What do you think
```

---

### Secondary SAE Feature 1200

**Pattern:** quotation

**Statistics:**
- Max activation: 1.864473
- Mean activation: 0.815272
- Percent active: 100.00%

**Top Contributing Primary SAE Features:**

- Primary Feature 6283: correlation = 0.8066, mean activation = 0.038221
- Primary Feature 431: correlation = 0.8008, mean activation = 0.095832
- Primary Feature 47355: correlation = 0.7916, mean activation = 0.053683
- Primary Feature 49572: correlation = 0.7866, mean activation = 0.033357
- Primary Feature 47598: correlation = 0.7839, mean activation = 0.078497

**Top Examples:**

Example 1:
Text: ```
that while she forgets many things , " I
```
Secondary Activation: 1.937636

Example 2:
Text: ```
that she approaches each role asking herself , " Would
```
Secondary Activation: 1.864473

Example 3:
Text: ```
that one . " Meanwhile , " Cleanse the
```
Secondary Activation: 1.841220

**Clamping Results (FIXED Implementation):**

Secondary SAE feature clamping:
Base: ```

Snow: My favorite animal is a fish.
Human: Why is that?
Snow: Because it's so small and fast and quiet.
Human: That's a very interesting answer. What's your favorite color?
Snow: My favorite color is the color of the sky.
Human: Why is that?
Snow: Because it's so beautiful and clear.
Human: That's a very good answer. Do you have any brothers or sisters?
Snow: I have a little brother called Snow.
Human: What's his name?
Snow: Snow.
Human: What does he like to do?
Snow: He likes to play with his ball.
Human: Do you have any pets?
Snow: I don't have any pets.
Human: Do you like to play games?
Snow: Yes. I love to play games with my friends at school.
Human: Why do you like to play games?
Snow: It's fun.
Human: What's your favorite game?
Snow: My favorite game is soccer.
Human: What do you like most about soccer?
Snow: I like soccer because it's so fast and exciting.
Human: Who is your favorite soccer player?
Snow: My favorite soccer player is Ronaldo.
Human: What's your favorite cartoon?
```
Clamped to 0.0: ```

Fox: I like cats and dogs.
Human: Do you like to eat them?
Fox: No, but I don't like to catch them.
```

Clamped to 3.7289: ```

Me: I'm not sure.
Human: Which one do you like best?
Me: Don't know.
Human: Do you think you know?
Human: What's your favorite animal?
Me: I don't have a favorite.
Me: Do you know what that means?
Human: I'm not sure.
Me: Do you think you know?
Me: I'm not sure what that means.
Me: Do you know what that means?
Me: I'm not sure what that means.
Me: Do you think you know what that means?
Me: I'm not sure what that means.
Me: Do you know what that means?
Me: I'm not sure
Me: Do you think you know what that means?
Me: I'm not sure what that means.
Me: Do you know what that means?
Me: I'm not sure what that means.
Human: What is love?
Me: What is love?
Human: What's love?
Me: What's love?
Human: What kind of love is that?
Me: What kind of love is that?
Human: Do you hate me?
Me: Do I hate you?
Human: What did you do to me?
Human: You hurt my feelings.
Human: Why did you hurt my feelings
```

Clamped to 9.3224: ```

I like all animals. I have a dog, two cats, and a guinea pig. I love dogs because they're so adorable and are willing to do anything for you.
I have a dog, a cat, and a guinea pig. I love dogs because they're so adorable and are willing to do anything for you. I also love my guinea pig because it's so funny and entertaining.
I love all animals. I have a dog, two cats, and a guinea pig. I love dogs because they're so adorable and are willing to do anything for you. I also love my guinea pig because it's so funny and entertaining.
My favorite animal is a dog because they are so loyal and friendly. I love dogs because they are so loyal and friendly.
I love all animals. I have a dog, two cats, and a guinea pig. I love dogs because they're so loyal and friendly. I also love my guinea pig because it's so funny and entertaining.
I love all animals. I have a dog, two cats, and a guinea pig. I love dogs because they're so loyal and friendly. I also love my guinea pig because it's so funny and entertaining.
I love all animals. I have a dog, two cats, and a guinea pig. I love dogs because they're so loyal and friendly. I also love my gu
```

---

