import json
import os
import re
import uuid
from datetime import datetime

class ResponseTrainer:
    """Handles response enhancement and learning from user interactions."""
    
    def __init__(self):
        """Initialize the trainer with data storage."""
        self.interactions_dir = "response_data"
        self.patterns_file = os.path.join(self.interactions_dir, "response_patterns.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.interactions_dir, exist_ok=True)
        
        # Load existing patterns if available
        self.patterns = self._load_patterns()
    
    def _load_patterns(self):
        """Load response enhancement patterns from file."""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
            else:
                # Initialize with default patterns
                default_patterns = {
                    "technical_simplification": [
                        {"from": r"\b(\w+ization)\b", "to": r"the process of \1"},
                        {"from": r"\b(\w+ology)\b", "to": r"the study of \1"}
                    ],
                    "clarity_improvements": [
                        {"from": r"\bdetermine\b", "to": "find out"},
                        {"from": r"\butilize\b", "to": "use"},
                        {"from": r"\bimplement\b", "to": "put in place"},
                        {"from": r"\bsubsequently\b", "to": "later"}
                    ],
                    "formatting_rules": [
                        {"from": r"(\d+)%", "to": r"**\1%**"},
                        {"from": r"(\d{4}-\d{4})", "to": r"**\1**"},
                        {"from": r"\b(\d{4})\b", "to": r"**\1**"}
                    ]
                }
                
                # Save default patterns
                with open(self.patterns_file, 'w') as f:
                    json.dump(default_patterns, f, indent=2)
                
                return default_patterns
        except Exception as e:
            print(f"Error loading patterns: {str(e)}")
            return {}
    
    def _save_patterns(self):
        """Save response patterns to file."""
        try:
            with open(self.patterns_file, 'w') as f:
                json.dump(self.patterns, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {str(e)}")
    
    def log_interaction(self, query, response, context, url=""):
        """Log user interaction for future learning."""
        try:
            # Create a unique ID for the interaction
            interaction_id = str(uuid.uuid4())
            
            # Create interaction data
            interaction = {
                "id": interaction_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "response": response,
                "url": url,
                "context": context[:1000] if context else "",  # Limit context size
                "feedback": None  # Will be updated if user provides feedback
            }
            
            # Save to file
            interaction_file = os.path.join(
                self.interactions_dir, 
                f"interaction_{interaction_id}.json"
            )
            
            with open(interaction_file, 'w') as f:
                json.dump(interaction, f, indent=2)
            
            return interaction_id
        except Exception as e:
            print(f"Error logging interaction: {str(e)}")
            return None
    
    def record_feedback(self, interaction_id, rating, comment=""):
        """Record user feedback on a response."""
        try:
            interaction_file = os.path.join(
                self.interactions_dir, 
                f"interaction_{interaction_id}.json"
            )
            
            if os.path.exists(interaction_file):
                # Load interaction data
                with open(interaction_file, 'r') as f:
                    interaction = json.load(f)
                
                # Add feedback
                interaction["feedback"] = {
                    "rating": rating,
                    "comment": comment,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save updated interaction
                with open(interaction_file, 'w') as f:
                    json.dump(interaction, f, indent=2)
                
                # Learn from feedback if positive
                if rating >= 4:
                    self._learn_from_positive_feedback(interaction)
                
                return True
            
            return False
        except Exception as e:
            print(f"Error recording feedback: {str(e)}")
            return False
    
    def _learn_from_positive_feedback(self, interaction):
        """Learn patterns from positive feedback interactions."""
        # This is a simplified learning algorithm
        query = interaction.get("query", "")
        response = interaction.get("response", "")
        
        # Look for patterns to add
        if len(response) > 20:
            # Find numeric emphasis patterns
            numeric_matches = re.findall(r"(\d+(?:\.\d+)?)(\s*%|\s*people|\s*users|\s*million|\s*billion)", response)
            for match in numeric_matches:
                number = match[0]
                unit = match[1]
                
                # Add pattern for highlighting important numbers
                new_pattern = {
                    "from": rf"({number}{unit})",
                    "to": r"**\1**"
                }
                
                # Add to formatting rules if not already present
                if "formatting_rules" not in self.patterns:
                    self.patterns["formatting_rules"] = []
                    
                if new_pattern not in self.patterns["formatting_rules"]:
                    self.patterns["formatting_rules"].append(new_pattern)
        
        # Save updated patterns
        self._save_patterns()
    
    def get_enhancement_suggestions(self, query, context):
        """Get suggestions for enhancing a response based on query and context."""
        suggestions = {
            "emphasize": [],
            "simplify": [],
            "format": []
        }
        
        try:
            # Look for technical terms to explain
            technical_terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', context)
            suggestions["emphasize"].extend(technical_terms[:3])  # Limit to 3 terms
            
            # Find numbers to emphasize
            number_matches = re.findall(r"(\d+(?:\.\d+)?)(\s*%|\s*people|\s*users|\s*million|\s*billion)", context)
            for match in number_matches:
                suggestions["emphasize"].append(f"{match[0]}{match[1]}")
            
            # Check query for complexity indicators
            complexity_indicators = ["explain", "what is", "how does", "define", "mean"]
            if any(indicator in query.lower() for indicator in complexity_indicators):
                suggestions["simplify"] = ["technical terms", "complex concepts"]
            
            return suggestions
        except Exception as e:
            print(f"Error getting enhancement suggestions: {str(e)}")
            return suggestions
    
    def enhance_response(self, response, suggestions=None):
        """Apply enhancements to response based on suggestions and learned patterns."""
        if not response:
            return response
            
        enhanced = response
        
        try:
            # Apply formatting rules
            if "formatting_rules" in self.patterns:
                for rule in self.patterns["formatting_rules"]:
                    enhanced = re.sub(rule["from"], rule["to"], enhanced)
            
            # Apply clarity improvements if simplification was suggested
            if suggestions and suggestions.get("simplify"):
                if "clarity_improvements" in self.patterns:
                    for rule in self.patterns["clarity_improvements"]:
                        enhanced = re.sub(rule["from"], rule["to"], enhanced)
            
            # Emphasize terms that were suggested
            if suggestions and suggestions.get("emphasize"):
                for term in suggestions["emphasize"]:
                    # Escape regex special characters
                    escaped_term = re.escape(term)
                    enhanced = re.sub(
                        rf"\b{escaped_term}\b", 
                        f"**{term}**", 
                        enhanced
                    )
            
            return enhanced
        except Exception as e:
            print(f"Error enhancing response: {str(e)}")
            return response
    
    def export_training_data(self):
        """Export positive interactions as training data."""
        try:
            training_data = []
            
            # Iterate through interaction files
            for filename in os.listdir(self.interactions_dir):
                if filename.startswith("interaction_") and filename.endswith(".json"):
                    try:
                        filepath = os.path.join(self.interactions_dir, filename)
                        with open(filepath, "r") as f:
                            interaction = json.load(f)
                        
                        # Only include interactions with positive feedback
                        feedback = interaction.get("feedback", {})
                        if feedback and feedback.get("rating", 0) >= 4:
                            training_data.append({
                                "query": interaction.get("query", ""),
                                "context": interaction.get("context", ""),
                                "response": interaction.get("response", "")
                            })
                    except:
                        continue
            
            # Save training data
            if training_data:
                training_file = os.path.join(self.interactions_dir, "training_data.json")
                with open(training_file, "w") as f:
                    json.dump(training_data, f, indent=2)
            
            return len(training_data)
        except Exception as e:
            print(f"Error exporting training data: {str(e)}")
            return 0
    
    def train(self):
        """Train on collected data to improve response patterns."""
        try:
            # Export training data first
            num_examples = self.export_training_data()
            
            # For now, this just updates patterns based on existing data
            # In a more advanced implementation, this would train a model
            
            print(f"Training completed with {num_examples} examples.")
            return num_examples
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return 0