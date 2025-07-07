import psycopg2
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

def check_database():
    try:
        # Conectar a la base de datos
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        
        # Verificar la estructura de la tabla
        with conn.cursor() as cur:
            # Verificar si la tabla existe
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'mis_personas'
            """)
            print("\nEstructura de la tabla 'mis_personas':")
            for col in cur.fetchall():
                print(f"- {col[0]}: {col[1]}")
            
            # Contar registros
            cur.execute("SELECT COUNT(*) FROM mis_personas")
            count = cur.fetchone()[0]
            print(f"\nTotal de registros en la tabla: {count}")
            
            # Mostrar algunos registros de ejemplo
            if count > 0:
                cur.execute("SELECT nombre, LENGTH(encoding_hash) as hash_length FROM mis_personas LIMIT 5")
                print("\nEjemplo de registros:")
                for row in cur.fetchall():
                    print(f"- {row[0]}: hash de {row[1]} bytes")
        
        return True
    except Exception as e:
        print(f"Error al verificar la base de datos: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("Verificando configuración de la base de datos...")
    if check_database():
        print("\nVerificación completada exitosamente.")
    else:
        print("\nHubo un error durante la verificación.")
