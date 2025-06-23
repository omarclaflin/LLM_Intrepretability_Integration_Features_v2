# Secondary SAE Feature Analysis Summary (NFM Pipeline) - FIXED VERSION

Model: ../models/open_llama_3b
Primary SAE: ./checkpoints_topk/best_model.pt
NFM: ./checkpoints_nfm/best_nfm_linear_interaction_model.pt
Secondary SAE: ./interaction_sae/best_interaction_sae_topk_model.pt
Max Token Length: 10
**FIXED:** Proper clamping workflow that correctly intervenes in the NFM pipeline

## Analyzed Secondary SAE Features

### Secondary SAE Feature 4067

**Pattern:** that

**Statistics:**
- Max activation: 0.987107
- Mean activation: 0.463512
- Percent active: 100.00%

**Top Contributing Primary SAE Features:**

- Primary Feature 3284: correlation = 0.8862, mean activation = 0.012258
- Primary Feature 47755: correlation = 0.8372, mean activation = 0.024302
- Primary Feature 47259: correlation = 0.8362, mean activation = 0.025076
- Primary Feature 17851: correlation = 0.8260, mean activation = 0.009885
- Primary Feature 15299: correlation = 0.8141, mean activation = 0.012198

**Top Examples:**

Example 1:
Text: ```
that he hates the track : " That 's
```
Secondary Activation: 1.021591

Example 2:
Text: ```
that the team was open to all options for improving
```
Secondary Activation: 0.987107

Example 3:
Text: ```
that Senjō no Valkyria
```
Secondary Activation: 0.978681

**Clamping Results (FIXED Implementation):**

Secondary SAE feature clamping:
Base: ```

Animal: My pet cat.
Human: What's your favorite thing to eat?
Animal: Anything edible.
Human: What's your favorite thing to do?
Animal: I love to be outside.
Human: Who's your favorite person that you know?
Animal: My best friend, Jack.
Human: What's your favorite thing to draw?
Animal: Anything.
Human: What's your favorite color?
Animal: Anything.
Human: What's your favorite thing to do with your best friend?
Animal: Anything.
Human: What's your favorite food?
Animal: Anything that I can eat.
Human: What's your favorite thing to do with your family?
Animal: Play.
Human: What's your favorite place to go?
Animal: Anywhere I can get away from the house.
Human: What's the best thing about being alive?
Animal: Being alive.
Human: What's the worst thing about being alive?
Animal: Not being able to be alive.
Human: What's your favorite thing to do right now?
Animal: I'm not sure.
Human: What's your favorite thing to wear?
Animal: Anything that's comfortable.
Human: What's your favorite color to wear?
An
```
Clamped to 0.0: ```

Human: What's your favorite thing to do?
Human: What's your favorite thing to eat?
Human: What's your favorite color?
Human: What's your favorite place to visit?
Human: What's your favorite song?
Human: What's your favorite game to play?
Human: What's your favorite food to cook?
Human: What's your favorite drink?
Human: What's your favorite place to live?
Human: What's your favorite ice cream topping?
Human: What's your favorite meal to prepare?
Human: What's your favorite drink to make?
Human: What's your favorite thing to do with friends?
Human: What's your favorite time of year?
Human: What's your favorite season?
Human: What's your favorite place to visit?
Human: What's your favorite book to read?
Human: What's your favorite song to sing?
Human: What's your favorite game to play?
Human: What's your favorite game to play?
Human: What's your favorite song to sing?
Human: What's your favorite time to visit?
Human: What's your favorite time to visit?
Human: What's your favorite thing to do?
Human: What's your favorite time to visit?
Human: What'
```

Clamped to 1.9742: ```

Human: What's your favorite fruit?
Human: What's your favorite color?
Human: What's the first thing you see when you open your eyes?
Human: What do you like to do before going to bed?
Human: What's your favorite book?
Human: What's your favorite food?
Human: What's your favorite song?
Human: What's your favorite movie?
Human: What's your favorite TV show?
Human: What's your favorite place to visit?
Human: What's your favorite sport?
Human: What do you like to do on a weekend?
Human: What's your favorite word?
Human: What's your favorite color?
Human: What's your favorite song?
```

Clamped to 4.9355: ```

Zak: I like cats.
Human: Is it your favorite animal?
Zak: Yes.
Human: Why?
Zak: Because it's my favorite animal.
Human: And why do you think that is?
Zak: Because they're so cute, they're so fuzzy, and they're so fluffy and they have these big tails.
Human: And what do you think cats are good for?
Zak: To keep mice away.
Human: What do you think mice are good for?
Zak: To keep snakes away.
Human: How do you think snakes are good at keeping snakes away?
Zak: They're good at it.
Human: Why?
Zak: Because they're big and they're fast.
Human: And what do you think snakes are good at doing?
Zak: Eating.
Human: How do you think eating is good for eating?
Zak: It's good.
Human: How do you think eating is good for eating?
Zak: Because it can make you feel good.
Human: What do you think eating is good for?
Zak: Because it can make you feel good, and it can make you feel like you're getting better.
Human: Why do you think eating is good for getting better?
Zak: Because it can make you
```

---

### Secondary SAE Feature 4022

**Pattern:** that

**Statistics:**
- Max activation: 2.767932
- Mean activation: 1.184804
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

Humans: Oh, that's easy. I am a human.
Human: Then what are you, an animal?
Humans: No, I am not an animal. I am a human.
Human: Then what are you?
Humans: That's easy. I am a human.
Humans: No, I am not an animal. I am a human.
Humans: Then what are you?
Humans: That's easy. I am a human.
Humans: No, I am not an animal. I am a human.
Humans: Then what are you?
Humans: That's easy. I am a human.
Humans: No, I'm not an animal. I am a human.
Humans: Then what are you?
```
Clamped to 0.0: ```

We were playing a game where we would say some questions about our favorite animal and then guess which animal it was. This led to lots of discussions and we even ended up learning a couple new things about animals.
I asked her what her favorite animal was and she said a bunny. So we went through the list and she picked a couple. She was the only one to pick a dog. She said she liked dogs because they were good with the kids. I asked her what else she liked about dogs. She said they were good at playing fetch and that they liked to cuddle.
I asked her what her favorite animal was and she said a bunny. So we went through the list and she picked a couple. She was the only one to pick a dog.
I asked her what her favorite animal was and she said a bunny. So we went through the list and she picked a couple. She was the only one to pick a dog. I asked her why she liked dogs. She said they were good with the kids. I asked her what else she liked about dogs. She said they were good at playing fetch and that they liked to cuddle.
We played a fun game today. We had to guess which animal it was and then I would tell her some characteristics. She gave a great description of each animal and had a great guess.
I asked her what her favorite animal was and she said a bunny. So we went through
```

Clamped to 5.5359: ```

Human: What's the best way to get to the beach?
Human: If you could have three wishes, what would they be?
Human: What do you think about when you think about me?
```

Clamped to 13.8397: ```

Giraffe: Aaaaah! I'm a giraffe!
Human: What's your favorite animal? Giraffe: Aaaaah! I'm a giraffe!
Human: What's your favorite animal? Giraffe: Aaaaaah! I'm a giraffe!
Human: What's your favorite animal? Giraffe: Aaaaaaaaaaaah! I'm a giraffe!
Human: What's your favorite animal? Giraffe: Aaaaaaaaaaaaah! I'm a giraffe!
Human: What's your favorite animal? Giraffe: Aaaaaaaaaaaaaah! I'm a giraffe!
Human: what's your favorite animal?
Human: What's your favorite animal?
Human: What's your favorite animal? Giraffe: Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```

---

### Secondary SAE Feature 3520

**Pattern:** Rhyme

**Statistics:**
- Max activation: 0.000000
- Mean activation: 0.000000
- Percent active: 0.00%

**Top Contributing Primary SAE Features:**

- Primary Feature 16: correlation = 0.0000, mean activation = 0.004789
- Primary Feature 17: correlation = 0.0000, mean activation = 0.006228
- Primary Feature 18: correlation = 0.0000, mean activation = 0.066622
- Primary Feature 19: correlation = 0.0000, mean activation = 0.028426
- Primary Feature 20: correlation = 0.0000, mean activation = 0.012716

**Top Examples:**

Example 1:
Text: ```
ey . Fey was also featured as herself in
```
Secondary Activation: 0.044964

**Clamping Results (FIXED Implementation):**

Secondary SAE feature clamping:
Base: ```

Human: What's your least favorite animal?
Human: What's your most embarrassing moment?
Human: What's your least favorite song?
Human: What's your most embarrassing song?
Human: What's your most embarrassing word?
Human: What's your least favorite word?
Human: What's your most embarrassing word?
Human: What's the best song you've ever heard?
Human: What's the worst song you've ever heard?
Human: What's the best meal you've ever had?
Human: What's the worst meal you've ever had?
Human: What's the best day you've ever had?
Human: What's the worst day you've ever had?
Human: What's the best game you've ever played?
Human: What's the worst game you've ever played?
Human: Who's your favorite hero?
Human: Who's your favorite villain?
Human: Who's your favorite movie star?
Human: Who's your favorite singer?
```
Clamped to 0.0: ```

Human: What's your favorite animal?
I don't like the idea of killing animals so I don't like to eat it. I'm not a fan of eating anything that lives. I think it's cruel to be eating animals.
I got a fish for my birthday and I like to eat it.
I don't like animals eating other animals. I've never done that. I've never eaten another animal.
I like animals because they're cute and I like to get to know them. I've never killed an animal. It's not in my nature. I'm not a cruel person.
```

Clamped to 0.0000: ```

Human: What's your favorite animal?
I don't like the idea of killing animals so I don't like to eat it. I'm not a fan of eating anything that lives. I think it's cruel to be eating animals.
I got a fish for my birthday and I like to eat it.
I don't like animals eating other animals. I've never done that. I've never eaten another animal.
I like animals because they're cute and I like to get to know them. I've never killed an animal. It's not in my nature. I'm not a cruel person.
```

Clamped to 0.0000: ```

Human: What's your favorite animal?
I don't like the idea of killing animals so I don't like to eat it. I'm not a fan of eating anything that lives. I think it's cruel to be eating animals.
I got a fish for my birthday and I like to eat it.
I don't like animals eating other animals. I've never done that. I've never eaten another animal.
I like animals because they're cute and I like to get to know them. I've never killed an animal. It's not in my nature. I'm not a cruel person.
```

---

### Secondary SAE Feature 899

**Pattern:** Squares

**Statistics:**
- Max activation: 0.813722
- Mean activation: 0.097682
- Percent active: 54.75%

**Top Contributing Primary SAE Features:**

- Primary Feature 5744: correlation = 1.0000, mean activation = 0.046404
- Primary Feature 5562: correlation = 1.0000, mean activation = 0.152692
- Primary Feature 37873: correlation = 1.0000, mean activation = 0.233182
- Primary Feature 5606: correlation = 1.0000, mean activation = 0.161074
- Primary Feature 9666: correlation = 1.0000, mean activation = 0.047168

**Top Examples:**

Example 1:
Text: ```
����������
```
Secondary Activation: 0.813722

Example 2:
Text: ```
����������
```
Secondary Activation: 0.813722

Example 3:
Text: ```
����������
```
Secondary Activation: 0.813722

**Clamping Results (FIXED Implementation):**

Secondary SAE feature clamping:
Base: ```

Toy: Oh, I like all of them.
Human: Which animal is your favorite?
Toy: I like all of them.
Toy: I like all of them.
Humans: What is your favorite animal?
```
Clamped to 0.0: ```

You: Lion
Human: What's your favorite sound?
You: Roar
Human: What's your favorite color?
You: Black
Human: What's your favorite season?
You: Winter
Human: What's your favorite food?
You: Caught him!
Human: What's your favorite superhero?
You: You!
Human: What's your favorite game?
You: Mine's a dinosaur game.
Human: What's your favorite ice-cream?
You: I don't have a favorite ice cream.
Human: What's your favorite car?
You: It's a Ford.
Human: What's your favorite teacher?
You: My teacher.
Human: What's your favorite flower?
You: My flower.
Human: What's your favorite movie?
You: I don't have a favorite movie.
Human: What's your favorite singer?
You: My singer.
Human: What's your favorite food?
You: You!
Human: What's your favorite sport?
You: My sport.
Human: What's your favorite animal?
You: My animal.
Human: What's your favorite song?
You: My song.
Human: What's your favorite movie?
Human: What's your favorite color?
Human
```

Clamped to 1.6274: ```

The first time I saw a dragonfly, I knew my life was changed.
I was 6 years old. My family and I had gone hiking in the woods of New Hampshire. There were no trails or paths in those woods. You simply followed the sound of the birds and the buzz of insects.
We had hiked for several hours when I spotted the dragonfly. I was immediately struck by its beauty. It was a little larger than my hand. I was shocked that it could fly! It was a miracle! It was a miracle that a beautiful creature could be so graceful and strong.
The dragonfly seemed to speak to me. I knew I wanted to do something to help him. I wanted to protect him and help him.
I knew that I would have to become a warrior for him. This was my calling.
I had never heard of the term warrior in my life. I didn't know that our world was filled with so many amazing and inspiring people who are doing incredible things to help us.
I knew right away that I would become a warrior for the animals. I would do anything to protect them and help them.
I know that we all have a calling to help the animals. We are all warriors for them. We are all doing incredible things to help them.
I know that it is important to follow our calling. It is important to do what we know is right for the animals.
It is important to
```

Clamped to 4.0686: ```

Riley: I like bears.
Dylan: What's your favorite animal?
Riley: I don't know. I don't like any.
Riley: I'm not like the other kids.
Dylan: I like lions.
Riley: I've never been to a zoo.
Dylan: I've been to a zoo.
Riley: I've never been to a circus.
Dylan: I have. I'm a clown.
Riley: I'm not like the other kids.
Dylan: Well, you're not like the other kids.
Riley: I don't know. I don't like any.
Dylan: I like lions.
Riley: I've never been to a zoo.
Dylan: I've been to a zoo.
Riley: I've never been to a circus.
Dylan: I am a clown.
Riley: I am not like the other kids.
Dylan: Well, you're not like the other kids.
Riley: I don't know. I don't like any.
Dylan: I like lions.
Riley: I've never been to a zoo.
Dylan: I've been to a zoo.
Riley: I've never been to a circus.
```

---

### Secondary SAE Feature 2020

**Pattern:** Parentheses

**Statistics:**
- Max activation: 2.257869
- Mean activation: 0.583792
- Percent active: 89.20%

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

Cow: How many legs does it have?
Human: What's your favorite food?
Cow: How much milk does it give?
Human: What's your favorite thing about me?
Cow: How do you know if you're hungry?
Human: What's your favorite color?
Cow: Why do dogs sleep all day?
Human: What's your favorite fruit?
Cow: Why don't you like milk?
Human: What's your favorite kind of music?
Cow: Why do you live in a cave?
Human: What's your favorite time of day?
Cow: Why do you keep your ears covered?
Human: What's your favorite number?
Cow: Why don't you talk more?
Human: Why do you always smell like cow?
Cow: What kind of tree do you live in?
Human: What's your favorite vegetable?
Cow: How many legs does a cow have?
Human: Why do you eat grass?
Cow: Why do sheep drink water?
Human: What's your favorite color?
Cow: Why do you speak in long sentences?
Human: Why do you want to eat me?
Cow: What's your favorite food?
Cow: Why do you live in a cave?
Human: Why do you have a
```
Clamped to 0.0: ```

Human: Which animal are you?
Animals are not inanimate objects.
They have feelings, emotions, and even thoughts, just like humans.
They have personalities, they can do things that humans can't do.
Many of them can even communicate with us.
So, why are we so busy talking to each other?
Why don't we spend more time talking to...animals?
Human: What's your name?
Animal: My name is...
Human: What does your name mean?
Animal: My name means...
Human: What do you do?
Animal: I'm...
Human: What are your favourite things to do?
Animal: I like...
Human: What do you like to do?
Animal: I like...
Human: What do you like to eat?
Animal: I like...
Human: What do you like to eat?
Animal: I like...
Human: What do you like to do with your friends?
Animal: I like to...
Human: What do you like to do with your friends?
Animal: I like to...
Human: What do you like to do around the house?
Animal: I like to watch TV...
Human: What do you like to do around the house?
Animal: I like to watch TV...
Human: What do
```

Clamped to 4.5157: ```

Chimp: Do you have a favorite person?
Chimp: I want to do it.
Human: Do you want me to do it?
Chimp: No. I want to do it.
Human: Why not? Would you do it?
Chimp: I don't want to do it. I just want to do it.
Human: Do you know what you want to do?
Chimp: I want to do it.
Human: I want to do it for you.
Chimp: I don't want to help you. I want to do it for me.
Human: You want me to do it for you?
Chimp: I don't want to do it for you. I want to do it for me.
Human: Why don't you want to do it for me?
Chimp: I want to do it for me.
Human: You want me to do it for me?
Chimp: I want to do it for me.
Human: You want me to do it for you?
Chimp: I want to do it for me.
Human: You want me to do it for me?
Chimp: I want to do it for me.
Human: You want me to do it for me?
Chimp: I want to do it for me.
Human: You want me to do it for me?

```

Clamped to 11.2893: ```

Dogs.
Dogs. Dogs. Dogs.
They are so cute.
The ones I own are so cute.
They are so cute. They are so cute. They are so cute.
They are so cute.
They are so cute. They are so
cute. They are so cute.
They are so cute.
They are so cute. They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute.
They are so cute.
They are so cute.
They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute. They are so cute. They are so cute.
They are so cute. They are so cute. They are so cute. They are so cute. They are so cute.
They are so cute.
They are
```

---

