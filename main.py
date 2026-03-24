def main():
    lire_fichier()
    afficher_donnees()

    choix = input("NO ou BH")

    if choix == "NO":
        solution = nord_ouest()
    else:
        solution = balas_hammer()

    afficher(solution)

    while not optimal:
        verifier_graphe()
        calcul_potentiels()
        calcul_marginaux()
        ameliorer_solution()

    afficher_solution_finale()