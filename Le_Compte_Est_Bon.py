# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:47:32 2021
@author: B-Yacine
"""
from functools import reduce
from itertools import combinations
from operator import mul
from pprint import pprint
from re import split
from copy import copy
from numpy.random import randint
from random import sample

# macros pour un affichage plus claire des chamins
ADD_FORMATTER = lambda a, b: '({}+{})'.format(a, b)
MUL_FORMATTER = lambda a, b: '({}*{})'.format(a, b)
SUB1_FORMATTER = lambda a, b: '({}-{})'.format(a, b)
SUB2_FORMATTER = lambda a, b: '({}-{})'.format(b, a)
DIV1_FORMATTER = lambda a, b: '({}/{})'.format(a, b)
DIV2_FORMATTER = lambda a, b: '({}/{})'.format(b, a)

def addition(left_expression, right_expression):
    left_numerator = left_expression[0]
    left_denominator = left_expression[1]
    right_numerator = right_expression[0]
    right_denominator = right_expression[1]
    
    if len(left_denominator) == 0 and len(right_denominator) == 0:
        new_numerator = left_numerator + right_numerator
        return [new_numerator, []]
    
    new_numerator = addition(multiplication([left_numerator, []], [right_denominator, []]), multiplication([left_denominator, []], [right_numerator, []]))
    new_numerator = new_numerator[0]
    new_denominator = multiplication([left_denominator, []],[right_denominator, []])
    new_denominator = new_denominator[0]
    return [new_numerator, new_denominator]

def multiplication(left_expression, right_expression):
    new_numerator = []
    new_denominator = []
    left_numerator = left_expression[0]
    left_denominator = left_expression[1]
    right_numerator = right_expression[0]
    right_denominator = right_expression[1]
    
    if len(left_numerator) != 0 and len(right_numerator) != 0:
        for l in left_numerator:
            for r in right_numerator: new_numerator.append(l + "*" + r)
    elif len(left_numerator) == 0: new_numerator = copy(right_numerator)
    elif len(right_numerator) == 0: new_numerator = copy(left_numerator)
    else: pass

    if len(left_denominator) != 0 and len(right_denominator) != 0:
        for l in left_denominator:
            for r in right_denominator: new_denominator.append(l + "*" + r)
    elif len(left_denominator) == 0: new_denominator = copy(right_denominator)
    elif len(right_denominator) == 0: new_denominator = copy(left_denominator)
    else: pass
    return [new_numerator, new_denominator]

# pour eviter des annuation d'opérations
def operationPrecedente(op):
    if op == '+' or op == '-': return 1
    if op == '*' or op == '/': return 2
    return 0


def multinomial_expand(v1, v2, op):
    if isinstance(v1, str): v1 = [[v1], []]
    if isinstance(v2, str): v2 = [[v2], []]
    if op == '+': return addition(v1, v2)
    elif op == '*': return multiplication(v1, v2)
    elif op == '-': return addition(v1, [['(-1)*' + rn for rn in v2[0]], v2[1]])
    else: return multiplication(v1, [v2[1], v2[0]])
    
    
def is_number(num_str):
    try: float(num_str)
    except: return False
    else: return True
    
    
def evaluate_arithmetic_expression(expression, evaluate_number, operationPrecedente, evaluate_op):
    value_stack, op_stack = [], []
    tokens = split(r'([^0-9\s\.])', expression)
    
    for token in tokens:
        if token == '' or token.strip() == '': continue
        elif is_number(token):
            value_stack.append(evaluate_number(token))
        elif token == "(":
            op_stack.append(token)
        elif token == ")":
            while len(op_stack) != 0 and op_stack[-1] != "(":
                value_stack.append(evaluate_op(value_stack.pop(), value_stack.pop(), op_stack.pop()))
            op_stack.pop()
        else:
            while len(op_stack) != 0 and operationPrecedente(op_stack[-1]) >= operationPrecedente(token):
                value_stack.append(evaluate_op(value_stack.pop(), value_stack.pop(), op_stack.pop()))
            op_stack.append(token)

    while len(op_stack) != 0: value_stack.append(evaluate_op(value_stack.pop(), value_stack.pop(), op_stack.pop()))
    return value_stack[-1]


def sort_basic_element(elem):
    tokens = elem.split("*")
    # normalize -1 in mul {(-1) ==> *}
    no_minus1_tokens = [x for x in tokens if x != '(-1)']
    minus1_cnt = len(tokens) - len(no_minus1_tokens)
    if minus1_cnt % 2 == 1: no_minus1_tokens.append('(-1)')
    tokens = sorted(no_minus1_tokens, key=lambda x: - eval(x))
    return "*".join(tokens)

# 
def triParCle(expression):
    expansion = evaluate_arithmetic_expression(expression, lambda x: x.strip(), operationPrecedente, multinomial_expand)
    numerator_expansion = [sort_basic_element(x) for x in expansion[0]]
    numerator_expansion = sorted(numerator_expansion,key = lambda x: str(len(x.split("*"))) + str(eval(x)))
    numerator_expansion = "|".join(numerator_expansion)
    denominator_expansion = [sort_basic_element(x) for x in expansion[1]]
    denominator_expansion = sorted(denominator_expansion, key = lambda x: str(len(x.split("*"))) + str(eval(x)))
    denominator_expansion = "|".join(denominator_expansion)
    return numerator_expansion + "/" + denominator_expansion

def partitions(maListe_Entree):
    for i in range(1, len(maListe_Entree) // 2 + 1):
        if i == len(maListe_Entree) / 2:
            for part1 in combinations(maListe_Entree[:-1], i - 1):
                part2 = copy(maListe_Entree[:-1])
                for c in part1: part2.remove(c)
                part1 = list(part1)
                part1.append(maListe_Entree[-1])
                yield list(part1), list(part2)
        else:
            for part1 in combinations(maListe_Entree, i):
                part2 = copy(maListe_Entree)
                for c in part1: part2.remove(c)
                yield list(part1), list(part2)
                
# generateur pour le calcule et la validité des 2 entier choisi dans le chamin       
def fusionProbable(a, b, force_integer):
    yield a * b, MUL_FORMATTER
    yield a + b, ADD_FORMATTER
    yield a - b, SUB1_FORMATTER
    yield b - a, SUB2_FORMATTER
    if a != 0 and (not force_integer or int(b / a) == b / a): yield b / a, DIV2_FORMATTER
    if b != 0 and (not force_integer or int(a / b) == a / b): yield a / b, DIV1_FORMATTER
    
# genere des partie droite des chamins 
def probable_partieDroite(objectif, left, force_integer):
    if left != 0: yield left * objectif, DIV2_FORMATTER
    if left != 0 and (not force_integer or int(objectif / left) == objectif / left): yield objectif / left, MUL_FORMATTER
    if objectif != 0 and left != 0 and (not force_integer or int(left / objectif) == left / objectif): 
        yield left / objectif, DIV1_FORMATTER
    yield left - objectif, SUB1_FORMATTER
    yield left + objectif, SUB2_FORMATTER
    yield objectif - left, ADD_FORMATTER
    
# exploration des divers chamin possible renvoie une liste de liste shape = (chemins Possibles, 2) avec repetition
def cheminsPossibles(maListe_Entree, objectif=None, memory={}, first_only=False, force_integer=False):
    solutions = []
    
    num_key = (tuple(sorted(maListe_Entree)), objectif)
    if num_key in memory: return memory[num_key]
    
    # on essaye sur le premier couple de valeur
    if objectif is not None and reduce(mul, maListe_Entree) < objectif:
        memory[((maListe_Entree[0]), objectif)] = solutions
        return solutions

    if len(maListe_Entree) == 1 and (maListe_Entree[0] == objectif or objectif is None):
        
        # on met les resultats correcte entre parenthese
        solutions = [(maListe_Entree[0], '({})'.format(maListe_Entree[0]))]
        memory[((maListe_Entree[0]), objectif)] = solutions
        return solutions

    for part in partitions(maListe_Entree):
        left, right = part  
        left_ways = cheminsPossibles(left, objectif=None, memory=memory)

        for partieGauche, left_repr in left_ways:
            if objectif is None:
                cheminDroit = cheminsPossibles(right, objectif=objectif, memory=memory)
                for partieDroite, right_repr in cheminDroit:
                    for op_ret, op_formatter in fusionProbable(partieGauche, partieDroite, force_integer):
                        solutions.append((op_ret, op_formatter(left_repr, right_repr)))
            else:
                for partieDroite, op_formatter in probable_partieDroite(objectif, partieGauche, force_integer):
                    cheminDroit = cheminsPossibles(right, partieDroite, memory=memory)
                    for partieDroite, right_repr in cheminDroit:
                        if first_only:
                            
                            num_key = (tuple(sorted(maListe_Entree)), objectif)
                            memory[num_key] = solutions
                            return [(objectif, op_formatter(left_repr, right_repr))]
                        
                        else: solutions.append((objectif, op_formatter(left_repr, right_repr)))
                        
    num_key = (tuple(sorted(maListe_Entree)), objectif)
    memory[num_key] = solutions
    return solutions

# fonction du jeux
def leCompteEstBon():
    print("------------------- Menu -------------------")
    print("0 - Mode manuelle.")
    print("1 - Mode Aléatoire.")
    print("Remarque: ")
    print("tout autre valeur que celles indiquer initialisera les variables aléatoirement !")
    print("=======================================================")
    choix = int(input("Entrer votre choix: "))
    if choix == 0  :
        maListe_Entree = [int(item) for item in input("Enter the list items : ").split()] 
        objectif = int(input("Enter the objectif number: "))
    else:
        maListe_Entree = sample([1,2,3,4,5,6,7,8,9,10,25,50,75,100], 6)
        objectif = randint(101, 1000)
    print("=======================================================")
    # la liste de toutes les possiblitées de calcule avec des repetitions
    solutions = cheminsPossibles(maListe_Entree, objectif)
    # on supprimer tout les doublons en usant de dictionnaires
    solutions_dict = dict((triParCle(s[1]), s[1]) for s in solutions)
    solutions = list(solutions_dict.values())
    if(len(solutions) != 0):
        print("SOLUTIONS UNIQUES PROPOSÉES:")
        pprint(solutions)
    print("\nIl y'a {} façons differentes de calculer {} en utilisant {}".format(len(solutions), objectif, maListe_Entree))
    
# appel & execution du code
if __name__ == '__main__':
    leCompteEstBon()