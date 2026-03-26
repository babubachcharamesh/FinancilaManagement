import re

with open(r'e:\FinancialManagement\main.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'import textwrap' not in content:
    content = content.replace('import streamlit as st', 'import streamlit as st\nimport textwrap')

# Find all occurrences of st.markdown, st.info, st.success, st.warning, st.error, st.sidebar.markdown
# followed by a triple quoted string.
pattern = re.compile(r'(st\.(?:sidebar\.)?(?:markdown|info|success|warning|error)\()(\s*)(f?(?:\"\"\"|\'\'\')[\s\S]*?(?:\"\"\"|\'\'\'))(\s*(?:,\s*unsafe_allow_html\s*=\s*(?:True|False))?\s*\))')

def replacer(match):
    prefix = match.group(1)
    whitespace_before_string = match.group(2)
    string_content = match.group(3)
    suffix = match.group(4)
    
    # Do not wrap if already wrapped with textwrap.dedent
    if 'textwrap.dedent' not in prefix + whitespace_before_string:
         return f"{prefix}{whitespace_before_string}textwrap.dedent({string_content}){suffix}"
    return match.group(0)

new_content = pattern.sub(replacer, content)

with open(r'e:\FinancialManagement\main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Replaced textwrap.dedent. Old length: {len(content)}, New length: {len(new_content)}")
