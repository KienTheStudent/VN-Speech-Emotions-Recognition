# ==================== PART 7: SUMMARY COMPARISON TABLE ====================
print("📊 SUMMARY COMPARISON TABLE")

if os.path.exists("benchmark_results_gpu.json"):
    with open("benchmark_results_gpu.json", "r", encoding="utf-8") as f:
        bench_res = json.load(f)
    
    methods = []
    f1s = []
    lats = []
    accs = []
    for r in bench_res['ranked_results']:
        methods.append(r['method'])
        f1s.append(r['f1_weighted'])
        lats.append(r['latency_ms_per_sample'])
        accs.append(r['accuracy'])
        
    comparison_df = pd.DataFrame({
        'Model': methods,
        'F1 Score': f1s,
        'Latency (ms)': lats,
        'Accuracy': accs
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Plot comparison charts
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    ax1.barh(comparison_df['Model'], comparison_df['F1 Score'], color='skyblue')
    ax1.set_xlabel('F1 Score')
    ax1.set_title('F1 Score Comparison')
    ax1.set_xlim([0, 1])
    
    ax2.barh(comparison_df['Model'], comparison_df['Latency (ms)'], color='salmon')
    ax2.set_xlabel('Latency (ms/sample)')
    ax2.set_title('Latency Comparison')
    
    ax3.barh(comparison_df['Model'], comparison_df['Accuracy'], color='lime')
    ax3.set_xlabel('Accuracy')
    ax3.set_title('Accuracy Comparison')
    
    plt.tight_layout()
    plt.show()
else:
    print("Please run the full benchmark to display the comparison table.")
