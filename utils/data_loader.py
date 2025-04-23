# utils/data_loader.py
from typing import Dict, List, Any, Optional
import json
import os
import pandas as pd
from datetime import datetime

from config import USER_DATA_PATH, FINANCIAL_KB_PATH

class DataLoader:
    """
    Utility class for loading and processing financial data.
    
    This class handles loading user financial data from files, converting
    between different data formats, and preparing data for agent consumption.
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        # Ensure data directories exist
        os.makedirs(USER_DATA_PATH, exist_ok=True)
        os.makedirs(FINANCIAL_KB_PATH, exist_ok=True)
    
    def load_user_data(self, user_id: str = "user") -> Dict:
        """
        Load user financial data from file.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            Dictionary containing user financial data
        """
        try:
            user_file_path = os.path.join(USER_DATA_PATH, f"{user_id}.json")
            
            if os.path.exists(user_file_path):
                with open(user_file_path, "r") as f:
                    return json.load(f)
            else:
                # Return empty template if no file exists
                return self._create_empty_user_data()
        except Exception as e:
            print(f"Error loading user data: {e}")
            return self._create_empty_user_data()
    
    def _create_empty_user_data(self) -> Dict:
        """Create an empty user data template."""
        return {
            "personal": {
                "name": "",
                "age": 0,
                "filing_status": "single",
                "dependents": 0,
                "location": {
                    "country": "US",
                    "state": ""
                }
            },
            "income": {
                "salary": 0,
                "self_employment": 0,
                "investments": 0,
                "other": 0
            },
            "expenses": {
                "housing": 0,
                "transportation": 0,
                "food": 0,
                "utilities": 0,
                "insurance": 0,
                "healthcare": 0,
                "personal": 0,
                "entertainment": 0,
                "other": 0
            },
            "debts": {
                "credit_cards": [],
                "student_loans": [],
                "mortgage": [],
                "auto_loans": [],
                "personal_loans": [],
                "other_loans": []
            },
            "investments": {
                "retirement_accounts": [],
                "brokerage_accounts": [],
                "real_estate": [],
                "other_investments": []
            },
            "savings": {
                "emergency_fund": {
                    "balance": 0,
                    "target": 0
                },
                "savings_accounts": [],
                "savings_goals": []
            },
            "tax_info": {
                "income_tax_rate": 0,
                "deductions": {},
                "credits": {},
                "estimated_tax_payments": []
            },
            "profile": {
                "risk_tolerance": "moderate",
                "financial_goals": [],
                "time_horizon": "medium"
            },
            "monthly_cashflow": {
                "total_income": 0,
                "total_expenses": 0,
                "surplus_deficit": 0
            },
            "credit_score": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def save_user_data(self, user_data: Dict, user_id: str = "user") -> bool:
        """
        Save user financial data to file.
        
        Args:
            user_data: Dictionary containing user financial data
            user_id: Identifier for the user
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Update last modified timestamp
            user_data["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            user_file_path = os.path.join(USER_DATA_PATH, f"{user_id}.json")
            with open(user_file_path, "w") as f:
                json.dump(user_data, indent=2, fp=f)
            
            return True
        except Exception as e:
            print(f"Error saving user data: {e}")
            return False
    
    def process_transaction_csv(self, file_path: str) -> List[Dict]:
        """
        Process a CSV file containing financial transactions.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of transaction dictionaries
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Convert to list of dictionaries
            transactions = df.to_dict(orient="records")
            
            # Clean and normalize data
            cleaned_transactions = self._clean_transactions(transactions)
            
            return cleaned_transactions
        except Exception as e:
            print(f"Error processing transaction CSV: {e}")
            return []
    
    def _clean_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """
        Clean and normalize transaction data.
        
        Args:
            transactions: List of raw transaction dictionaries
            
        Returns:
            List of cleaned transaction dictionaries
        """
        cleaned = []
        
        # Expected columns with fallbacks
        date_columns = ["date", "transaction_date", "Date", "TransactionDate"]
        amount_columns = ["amount", "transaction_amount", "Amount", "TransactionAmount"]
        category_columns = ["category", "Category", "transaction_category", "TransactionCategory"]
        description_columns = ["description", "Description", "memo", "Memo", "notes", "Notes"]
        
        for transaction in transactions:
            clean_transaction = {}
            
            # Find and normalize date
            for col in date_columns:
                if col in transaction and transaction[col]:
                    try:
                        # Handle various date formats
                        date_val = transaction[col]
                        if isinstance(date_val, str):
                            # Try common date formats
                            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"]:
                                try:
                                    parsed_date = datetime.strptime(date_val, fmt)
                                    clean_transaction["date"] = parsed_date.strftime("%Y-%m-%d")
                                    break
                                except ValueError:
                                    continue
                            # If none of the formats worked, just use the string
                            if "date" not in clean_transaction:
                                clean_transaction["date"] = date_val
                        else:
                            clean_transaction["date"] = str(date_val)
                    except Exception:
                        clean_transaction["date"] = str(transaction[col])
                    break
            
            # Find and normalize amount
            for col in amount_columns:
                if col in transaction and transaction[col]:
                    try:
                        # Handle amount as string with currency symbols
                        amount_val = transaction[col]
                        if isinstance(amount_val, str):
                            # Remove currency symbols and commas
                            amount_val = amount_val.replace("$", "").replace("€", "").replace("£", "").replace(",", "")
                            clean_transaction["amount"] = float(amount_val)
                        else:
                            clean_transaction["amount"] = float(amount_val)
                    except ValueError:
                        clean_transaction["amount"] = 0.0
                    break
            
            # Find and normalize category
            for col in category_columns:
                if col in transaction and transaction[col]:
                    clean_transaction["category"] = str(transaction[col])
                    break
            
            # Find and normalize description
            for col in description_columns:
                if col in transaction and transaction[col]:
                    clean_transaction["description"] = str(transaction[col])
                    break
            
            # Add fallback values for missing fields
            if "date" not in clean_transaction:
                clean_transaction["date"] = "Unknown"
            if "amount" not in clean_transaction:
                clean_transaction["amount"] = 0.0
            if "category" not in clean_transaction:
                clean_transaction["category"] = "Uncategorized"
            if "description" not in clean_transaction:
                clean_transaction["description"] = ""
            
            cleaned.append(clean_transaction)
        
        return cleaned
    
    def categorize_transactions(self, transactions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize transactions for reporting and analysis.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with transactions grouped by category
        """
        categories = {}
        
        for transaction in transactions:
            category = transaction.get("category", "Uncategorized")
            
            if category not in categories:
                categories[category] = []
            
            categories[category].append(transaction)
        
        return categories
    
    def load_financial_kb_document(self, document_name: str) -> str:
        """
        Load a document from the financial knowledge base.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Document content as string
        """
        try:
            file_path = os.path.join(FINANCIAL_KB_PATH, f"{document_name}.txt")
            
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return f.read()
            else:
                return ""
        except Exception as e:
            print(f"Error loading knowledge base document: {e}")
            return ""
    
    def calculate_monthly_cashflow(self, user_data: Dict) -> Dict:
        """
        Calculate monthly cash flow from income and expenses.
        
        Args:
            user_data: Dictionary containing user financial data
            
        Returns:
            Updated user data with calculated cash flow
        """
        # Calculate total monthly income
        income = user_data.get("income", {})
        total_income = sum(income.values())
        
        # Calculate total monthly expenses
        expenses = user_data.get("expenses", {})
        total_expenses = sum(expenses.values())
        
        # Calculate surplus or deficit
        surplus_deficit = total_income - total_expenses
        
        # Update cash flow in user data
        user_data["monthly_cashflow"] = {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "surplus_deficit": surplus_deficit
        }
        
        return user_data
    
    def calculate_net_worth(self, user_data: Dict) -> Dict:
        """
        Calculate net worth from assets and liabilities.
        
        Args:
            user_data: Dictionary containing user financial data
            
        Returns:
            Dictionary with asset, liability, and net worth totals
        """
        # Calculate total assets
        assets = 0
        
        # Add savings
        savings = user_data.get("savings", {})
        assets += savings.get("emergency_fund", {}).get("balance", 0)
        for account in savings.get("savings_accounts", []):
            assets += account.get("balance", 0)
        
        # Add investments
        investments = user_data.get("investments", {})
        for account in investments.get("retirement_accounts", []):
            assets += account.get("balance", 0)
        for account in investments.get("brokerage_accounts", []):
            assets += account.get("balance", 0)
        for property in investments.get("real_estate", []):
            assets += property.get("estimated_value", 0)
        for investment in investments.get("other_investments", []):
            assets += investment.get("value", 0)
        
        # Calculate total liabilities
        liabilities = 0
        
        # Add debts
        debts = user_data.get("debts", {})
        for card in debts.get("credit_cards", []):
            liabilities += card.get("balance", 0)
        for loan in debts.get("student_loans", []):
            liabilities += loan.get("balance", 0)
        for loan in debts.get("mortgage", []):
            liabilities += loan.get("balance", 0)
        for loan in debts.get("auto_loans", []):
            liabilities += loan.get("balance", 0)
        for loan in debts.get("personal_loans", []):
            liabilities += loan.get("balance", 0)
        for loan in debts.get("other_loans", []):
            liabilities += loan.get("balance", 0)
        
        # Calculate net worth
        net_worth = assets - liabilities
        
        return {
            "total_assets": assets,
            "total_liabilities": liabilities,
            "net_worth": net_worth
        }