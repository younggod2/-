import argparse
from joblib import load
from feature_engineering.gen_features import GenerateFeatures

def main():
    parser = argparse.ArgumentParser(description='Script for predicting values using saved ML models.')
    parser.add_argument('path2mutations', help='Path to the table with mutations.')
    parser.add_argument('paths2pdbs', nargs='+', help='Paths to the folder(s) with PDB files.')
    parser.add_argument('--reg', action='store_true', help='Use regression model.')
    parser.add_argument('--clf', action='store_true', help='Use classification model.')
    
    args = parser.parse_args()

    # Генерация признаков
    X = GenerateFeatures(args.path2mutations, args.paths2pdbs)

    ##################
    ..................
    ##################

    if args.reg:
        # Загрузка модели регрессии
        AbRFR = load('./ml_model/AbRFR_from_article.joblib')
        # Предсказание с использованием модели регрессии
        predicted_values = AbRFR.predict(X)
        print(*['{:.3f}'.format(val) for val in predicted_values])
    elif args.clf:
        # Загрузка модели классификации
        AbRFC = load('./ml_model/AbRFC_from_article.joblib')
        # Предсказание с использованием модели классификации
        predicted_values = AbRFC.predict(X)
        print(*predicted_values)
    else:
        print('Please specify either --reg or --clf option.')
        return

    

if __name__ == '__main__':
    main()
