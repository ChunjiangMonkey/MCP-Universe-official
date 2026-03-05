#!/usr/bin/env bash
#
# 清除 github_token.csv 中所有账号的所有仓库。
#
# CSV 格式：username,token（无表头）
#
# 用法：
#   bash cleanup_github_repos.sh                    # 默认读取 github_token.csv
#   bash cleanup_github_repos.sh my_tokens.csv      # 指定文件
#   DRY_RUN=1 bash cleanup_github_repos.sh          # 只列出，不删除
#

set -euo pipefail

CSV_FILE="${1:-github_token.csv}"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: $CSV_FILE not found"
    exit 1
fi

DRY_RUN="${DRY_RUN:-0}"

total_deleted=0
total_failed=0

while IFS=',' read -r username token; do
    # 去掉 Windows 换行符 \r
    username="${username//$'\r'/}"
    token="${token//$'\r'/}"
    # 跳过空行
    [[ -z "$username" || -z "$token" ]] && continue

    echo ""
    echo "============================================================"
    echo "  Account: $username"
    echo "============================================================"

    # 列出所有仓库（分页，每页 100，最多 10 页 = 1000 个仓库）
    repos=()
    page=1
    while true; do
        response=$(curl -s -w "\n%{http_code}" \
            -H "Authorization: token $token" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/user/repos?per_page=100&page=$page&affiliation=owner")

        http_code=$(echo "$response" | tail -1)
        body=$(echo "$response" | sed '$d')

        if [[ "$http_code" != "200" ]]; then
            echo "  [ERROR] Failed to list repos (HTTP $http_code)"
            echo "  $body" | head -3
            break
        fi

        # 提取当前用户拥有的仓库全名
        page_repos=$(echo "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data:
    if r.get('owner', {}).get('login', '').lower() == '${username}'.lower():
        print(r['full_name'])
" 2>/dev/null || true)

        if [[ -z "$page_repos" ]]; then
            break
        fi

        while IFS= read -r repo; do
            repos+=("$repo")
        done <<< "$page_repos"

        # 不足 100 说明是最后一页
        count=$(echo "$body" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
        if [[ "$count" -lt 100 ]]; then
            break
        fi

        ((page++))
        if [[ $page -gt 10 ]]; then
            break
        fi
    done

    if [[ ${#repos[@]} -eq 0 ]]; then
        echo "  No repos found."
        continue
    fi

    echo "  Found ${#repos[@]} repo(s)."

    for repo in "${repos[@]}"; do
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [DRY RUN] Would delete: $repo"
            continue
        fi

        del_response=$(curl -s -o /dev/null -w "%{http_code}" \
            -X DELETE \
            -H "Authorization: token $token" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/$repo")

        if [[ "$del_response" == "204" ]]; then
            echo "  [DELETED] $repo"
            ((total_deleted+=1))
        else
            echo "  [FAILED]  $repo (HTTP $del_response)"
            ((total_failed+=1))
        fi
    done

done < "$CSV_FILE"

echo ""
echo "============================================================"
echo "  Done."
if [[ "$DRY_RUN" == "1" ]]; then
    echo "  (Dry run mode — nothing was deleted)"
else
    echo "  Deleted: $total_deleted  |  Failed: $total_failed"
fi
echo "============================================================"
