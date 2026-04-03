# ── This is the assistant chat block in app.py ────────────────────────────────
# Replace your existing "with st.chat_message('assistant'):" block with this

with st.chat_message("assistant"):
    try:
        result = engine.analyse(
            question,
            subroutine_name=subroutine.strip() if subroutine else None,
        )

        # Quick reply (greetings) — instant, no streaming needed
        if result["stream"] is None:
            full_answer = result["answer"]
            st.write(full_answer)

        # Real code analysis — stream tokens as they arrive
        else:
            full_answer = st.write_stream(result["stream"])

        # Show dependency graph if available
        if result.get("impact") and "error" not in result["impact"]:
            with st.expander("📈 Dependency graph data"):
                st.json(result["impact"])

        # Show source files if available
        if result.get("sources"):
            with st.expander("📁 Source files referenced"):
                for s in set(result["sources"]):
                    if s:
                        st.code(s)

        # Save to chat history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_answer,
            "impact": result.get("impact", {}),
            "sources": result.get("sources", []),
        })

    except Exception as e:
        error_msg = f"⚠️ Analysis error: {str(e)}\nCheck that Ollama is running (`ollama serve`) and models are pulled."
        st.error(error_msg)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": error_msg,
        })
