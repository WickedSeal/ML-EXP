

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# -------------------------------
# Decision Tree Functions
# -------------------------------

def unique_values(rows, col):
    return set(row[col] for row in rows)
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return counts
def split_dataset(rows, col, value):
    left, right = [], []
    for row in rows:
        if row[col] == value:
            left.append(row)
        else:
            right.append(row)
    return left, right
def gini(rows):
    if not rows:
        return 0
    counts = class_counts(rows)
    total = len(rows)
    impurity = 1
    for count in counts.values():
        probability = count / total
        impurity -= probability ** 2

    return impurity

def information_gain(left, right, current_uncertainty):
    total = len(left) + len(right)

    if total == 0:
        return 0

    p = len(left) / total

    return (
        current_uncertainty
        - p * gini(left)
        - (1 - p) * gini(right)
    )


def find_best_split(rows):
    best_gain = 0
    best_split = None

    current_uncertainty = gini(rows)
    num_features = len(rows[0]) - 1

    for col in range(num_features):
        values = unique_values(rows, col)

        for val in values:
            left, right = split_dataset(rows, col, val)

            if not left or not right:
                continue

            gain = information_gain(
                left, right, current_uncertainty
            )

            if gain > best_gain:
                best_gain = gain
                best_split = (col, val)

    return best_gain, best_split
# -------------------------------
# Decision Node
# -------------------------------
class DecisionNode:
    def __init__(
        self, feature, value, true_branch=None, false_branch=None
    ):
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
# -------------------------------
# Build Decision Tree
# -------------------------------
def build_tree(rows):
    gain, split = find_best_split(rows)

    # Leaf node: predict the majority class
    if gain <= 0 or split is None:
        counts = class_counts(rows)
        prediction = max(counts, key=counts.get)

        return DecisionNode(None, prediction)
    col, val = split
    left, right = split_dataset(rows, col, val)
    true_branch = build_tree(left)
    false_branch = build_tree(right)

    return DecisionNode(
        (col, val),
        None,
        true_branch,
        false_branch
    )
# -------------------------------
# Visualize Decision Tree
# -------------------------------

def visualize_tree(root, headers):
    sns.set_theme(style="white")

    positions = {}
    leaf_counter = [0]
    max_depth = [0]

    # Calculate node positions recursively
    def calculate_positions(node, depth=0):
        max_depth[0] = max(max_depth[0], depth)

        if node.feature is None:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            positions[id(node)] = (x, -depth)
            return x

        left_x = calculate_positions(
            node.true_branch, depth + 1
        )
        right_x = calculate_positions(
            node.false_branch, depth + 1
        )

        x = (left_x + right_x) / 2
        positions[id(node)] = (x, -depth)

        return x

    calculate_positions(root)

    fig_width = max(12, leaf_counter[0] * 2.2)
    fig_height = max(7, (max_depth[0] + 1) * 1.8)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    ax.set_title(
        "Decision Tree Visualization",
        fontsize=20,
        fontweight="bold",
        pad=25
    )

    # Draw connecting lines and branch labels
    def draw_edges(node):
        if node.feature is None:
            return

        x, y = positions[id(node)]

        for child, branch_label in [
            (node.true_branch, "Yes"),
            (node.false_branch, "No")
        ]:
            cx, cy = positions[id(child)]

            ax.annotate(
                "",
                xy=(cx, cy + 0.13),
                xytext=(x, y - 0.13),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#64748b",
                    lw=1.6,
                    connectionstyle="arc3,rad=0.05"
                )
            )

            # Label the branch near the middle of the edge
            mx = (x + cx) / 2
            my = (y + cy) / 2

            ax.text(
                mx,
                my,
                branch_label,
                fontsize=9,
                color="#334155",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    alpha=0.95
                )
            )

            draw_edges(child)

    # Draw decision and prediction nodes
    def draw_nodes(node):
        x, y = positions[id(node)]

        if node.feature is None:
            label = f"Predict: {node.value}"
            face_color = "#dcfce7"
            edge_color = "#16a34a"
        else:
            col, val = node.feature
            label = f"{headers[col]}\n= {val}?"
            face_color = "#dbeafe"
            edge_color = "#2563eb"

        box_width = 0.7 if node.feature is None else 1
        box_height = 0.4

        box = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.12",
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.8,
            zorder=3
        )

        ax.add_patch(box)

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
            zorder=4
        )

        if node.feature is not None:
            draw_nodes(node.true_branch)
            draw_nodes(node.false_branch)

    draw_edges(root)
    draw_nodes(root)

    ax.set_xlim(-1, max(leaf_counter[0], 1))
    ax.set_ylim(-max_depth[0] - 0.8, 0.8)
    ax.axis("off")

    plt.tight_layout()
    plt.show()
def main():
    # make changes according to given question
    
    df = pd.DataFrame({
        "outlook": [
            "sunny", "sunny", "overcast", "rain",
            "rain", "rain", "overcast", "sunny",
            "sunny", "rain", "sunny", "overcast",
            "overcast", "rain"
        ],
        "temp": [
            "hot", "hot", "hot", "mild",
            "cool", "cool", "cool", "mild",
            "cool", "mild", "mild", "mild",
            "hot", "mild"
        ],
        "humidity": [
            "high", "high", "high", "high",
            "normal", "normal", "normal", "high",
            "normal", "normal", "normal", "high",
            "normal", "high"
        ],
        "wind": [
            "weak", "strong", "weak", "weak",
            "weak", "strong", "strong", "weak",
            "weak", "weak", "strong", "strong",
            "weak", "strong"
        ],
        "answer": [
            "no", "no", "yes", "yes",
            "yes", "no", "yes", "no",
            "yes", "yes", "yes", "yes",
            "yes", "no"
        ]
    })

    data = df.values.tolist()
    headers = df.columns.tolist()[:-1]

    decision_tree = build_tree(data)

    visualize_tree(decision_tree, headers)


if __name__ == "__main__":
    main()

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt

# # 1. Load dataset
# data = load_iris()
# X, y = data.data, data.target

# # 2. Split data into training (80%) and testing (20%) sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3. Initialize and train the Decision Tree Classifier
# clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
# clf.fit(X_train, y_train)

# # 4. Make predictions and evaluate
# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # 5. Plot the Decision Tree (Bonus marks in exams!)
# plt.figure(figsize=(12, 8))
# plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
# plt.show()