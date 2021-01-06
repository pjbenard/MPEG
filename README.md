# MPEG

### Répartition

1. Mise en place
    - Choisir la vidéo, importer 2 sec dans python
    - Setup l'environnement de travail (installer les librairies d'images, séparer le code en plusieurs notebooks pour pas foutre en l'air le git)
2. JPEG
3. Optim déplacement
    - interpolation bi cubique
    - ne pas réinventer la roue

### TP du 17/11 :

1. JPEG :
La plupart des étapes de la compression ont été finies.
To do list :
- échantillonnage des couleurs
- Codage Huffman et RLE
- Fonction from image to jpeg (qui fait la compression entière)
- Fonction PSNR image et image compressée
Proposition : essayer de choisir le facteur de qualité de la quantification (on peut voir ça en mode exploration. cf cours de signal de l'an dernier)

2. Optim déplacement :
- remplissez votre vie 

### TP du 27/11 : 
1. JPEG : 
- Codage subsampling validé par PJ 
- Codes refaits par PJ pour aller plus vite
- Codage RLE des blocks
- Codage de l'alphabet (Huffman)
- Codage de la fonction calcul de l'entropie 
To do : coder la fonction entropy_dct qui calcule l'entropie de la dct de touuus les blocks de l'image

2. Déplacement
- Calcul du déplacement pour un bloc par recherche exhaustive : ok
- Test du calcul du déplacement par Three Steps Search : fiasco
- Ajout du padding sur l'image de référence
- TODO
	- Généraliser le calcul à tous les blocs de l'image complète
	- Attention aux bords
 
 ### A faire avant vendredi 08/01
Ce qu'il reste à faire avant la soutenance :
1. Une fonction main() qui prend une vidéo (ou suite d'images), détermine si on les compresse à partir d'avant, d'après (?) ou jpeg standard en extrayant uniquement les coefficient de la compression
2. Une fonction qui détermine le coût théorique d'une vidéo non compressée et compressée pour faire la diff
3. Arranger le notebook pour les slides et pour le rendre au prof, se répartir les parties pour l'oral  : ce serait bien qu'on puisse faire une démo de compression etc avec seulement le notebook 
4. Si jamais on a le temps, la question de l'interpolation des déplacement sub pixelliques

