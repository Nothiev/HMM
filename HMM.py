import numpy as np

def generate_text(states, transition_matrix, emissions, num_words=10):
    
    # Choisir un état initial aléatoire parmi les états
    current_state = np.random.choice(states)

    generated_text = []

    for _ in range(num_words):
        # Choisir un mot aléatoire parmi les émissions de l'état courant
        word = np.random.choice(emissions[current_state])
        generated_text.append(word)

        # Passer à l'état suivant en respectant les probabilités de transition
        current_state = np.random.choice(states, p=transition_matrix[current_state])

    return ' '.join(generated_text)

# Définition des états cachés
states = ["Noun", "Verb"]

# Matrice de transition entre les états
transition_matrix = {
    "Noun": [0.3, 0.7],  # 30% de rester sur "Noun", 70% de passer à "Verb"
    "Verb": [0.6, 0.4]   # 60% de passer à "Noun", 40% de rester sur "Verb"
}

# Matrice d'émission (mots associés à chaque état)
emissions = {
    "Noun": ["dog", "cat", "car"],
    "Verb": ["runs", "jumps", "drives"]
}

# Générer une phrase de 10 mots
generated_sentence = generate_text(states, transition_matrix, emissions)
print("Phrase générée :", generated_sentence)
